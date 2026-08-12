#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transfer Learning — Step 2: does shared abiotic-stress biology transfer?
========================================================================
Pretrain the CARCM encoder on the LARGE drought corpus (44.5k rows), then
fine-tune on the small salt set, and compare against salt-from-scratch.

Three conditions, all on the SAME salt train/dev/test split (seed 42) and the
SAME shared feature space (so the only thing that changes is the initialisation):

  (A) scratch          — CARCM trained on salt only (no drought) = our baseline
  (B) transfer_full    — pretrain on drought, fine-tune ALL weights on salt
  (C) transfer_frozen  — pretrain on drought, FREEZE encoder, train only the
                         output head on salt (linear probe → pure representation
                         transfer)

Each condition is a `--num_seeds`-model ensemble (mean of sigmoid outputs).
The model architecture is identical to rice_classifier_v7b_salt (copied here so
this folder is fully self-contained). Winning HP config: lr=1e-4, hidden=128,
dropout=0.5, batch=128.

Outputs (transfer_learning/results/):
    tl_comparison.csv / tl_comparison.json   — metrics per condition
    tl_comparison.png                         — bar chart
    tl_predictions_<cond>.csv                 — salt-test predictions per condition
"""
import os
import sys
import json
import copy
import time
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── args / GPU (before torch) ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_seeds", type=int, default=5)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--finetune_lr", type=float, default=1e-4)   # winning config
parser.add_argument("--scratch_lr", type=float, default=1e-4)
parser.add_argument("--pretrain_lr", type=float, default=3e-4)
parser.add_argument("--pretrain_epochs", type=int, default=100)
parser.add_argument("--pretrain_patience", type=int, default=12)
parser.add_argument("--finetune_epochs", type=int, default=150)
parser.add_argument("--finetune_patience", type=int, default=25)
parser.add_argument("--weight_decay", type=float, default=0.01)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, average_precision_score,
)
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/transfer_learning/data"
RESULTS_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/transfer_learning/results"
CKPT_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/transfer_learning/checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

RANDOM_SEED = 42
NUM_BINS = 5


# =============================================================================
# MODEL  (copied verbatim from rice_classifier_v7b_salt — kept self-contained)
# =============================================================================
class ContextEmbeddingModule(nn.Module):
    def __init__(self, n_varieties, n_tissues, n_stages, embed_dim=64):
        super().__init__()
        self.variety_embed = nn.Embedding(n_varieties, embed_dim)
        self.tissue_embed = nn.Embedding(n_tissues, embed_dim)
        self.stage_embed = nn.Embedding(n_stages, embed_dim)
        nn.init.normal_(self.variety_embed.weight, 0, 0.1)
        nn.init.normal_(self.tissue_embed.weight, 0, 0.1)
        nn.init.normal_(self.stage_embed.weight, 0, 0.1)

    def forward(self, variety_idx, tissue_idx, stage_idx):
        return (self.variety_embed(variety_idx.squeeze(-1)),
                self.tissue_embed(tissue_idx.squeeze(-1)),
                self.stage_embed(stage_idx.squeeze(-1)))


class PositionalMotifProcessor(nn.Module):
    def __init__(self, n_motifs, n_bins, hidden_dim, dropout=0.4):
        super().__init__()
        self.n_motifs = n_motifs
        self.n_bins = n_bins
        self.bin_conv = nn.Sequential(
            nn.Conv1d(n_motifs, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bin_attention = nn.Sequential(nn.Linear(n_bins, n_bins), nn.Softmax(dim=-1))
        self.pos_fc = nn.Sequential(
            nn.Linear(n_motifs * 4, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

    def forward(self, binned, relpos, firstpos, lastpos, gradient):
        bs = binned.shape[0]
        binned_3d = binned.view(bs, self.n_motifs, self.n_bins)
        binned_weighted = binned_3d * self.bin_attention(binned_3d)
        bin_features = self.bin_conv(binned_weighted).mean(dim=2)
        pos_concat = torch.cat([relpos, firstpos, lastpos, gradient], dim=1)
        pos_features = self.pos_fc(pos_concat)
        return self.combine(torch.cat([bin_features, pos_features], dim=1))


class ContextEnhancedGatedFusionV5(nn.Module):
    def __init__(self, n_motifs, n_bins, n_cooc, n_varieties, n_tissues, n_stages,
                 num_seq_stats=4, hidden_dim=128, embed_dim=64, dropout=0.4,
                 n_vt=0, n_ts=0, n_vs=0):
        super().__init__()
        self.n_motifs = n_motifs
        self.n_bins = n_bins
        self.context_embed = ContextEmbeddingModule(n_varieties, n_tissues, n_stages, embed_dim)
        self.positional_proc = PositionalMotifProcessor(n_motifs, n_bins, hidden_dim, dropout)
        self.cooc_fc = nn.Sequential(
            nn.Linear(n_cooc, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2))
        self.count_embed = nn.Linear(1, 32)
        self.lstm = nn.LSTM(32, hidden_dim // 2, num_layers=2, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.count_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        context_input_dim = embed_dim * 3 + n_vt + n_ts + n_vs + num_seq_stats
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        motif_feature_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        self.context_query = nn.Linear(hidden_dim, hidden_dim)
        self.motif_key = nn.Linear(motif_feature_dim, hidden_dim)
        self.motif_value = nn.Linear(motif_feature_dim, hidden_dim)
        self.attn_scale = hidden_dim ** 0.5
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.dropout = nn.Dropout(dropout)

    def encoder_parameters(self):
        """All params EXCEPT the final output head (for linear-probe freezing)."""
        out_ids = {id(p) for p in self.output.parameters()}
        return [p for p in self.parameters() if id(p) not in out_ids]

    def forward(self, binned, relpos, firstpos, lastpos, gradient, count, cooc,
                seq_stats, variety, tissue, stage,
                variety_idx=None, tissue_idx=None, stage_idx=None,
                variety_tissue=None, tissue_stage=None, variety_stage=None, **kwargs):
        variety_emb, tissue_emb, stage_emb = self.context_embed(variety_idx, tissue_idx, stage_idx)
        pos_feat = self.positional_proc(binned, relpos, firstpos, lastpos, gradient)
        cooc_feat = self.cooc_fc(cooc)
        count_3d = self.count_embed(count.unsqueeze(-1))
        _, (h_n, _) = self.lstm(count_3d)
        count_feat = torch.relu(self.count_proj(torch.cat((h_n[-2], h_n[-1]), dim=1)))
        motif_combined = torch.cat([pos_feat, cooc_feat, count_feat], dim=1)
        context_parts = [variety_emb, tissue_emb, stage_emb, seq_stats]
        if variety_tissue is not None:
            context_parts.extend([variety_tissue, tissue_stage, variety_stage])
        context_feat = self.context_fc(torch.cat(context_parts, dim=1))
        query = self.context_query(context_feat)
        key = self.motif_key(motif_combined)
        value = self.motif_value(motif_combined)
        attn_weight = torch.sigmoid((query * key).sum(dim=-1, keepdim=True) / self.attn_scale)
        context_aware_motif = attn_weight * value
        motif_final = self.dropout(pos_feat) + context_aware_motif
        gate_weights = self.gate(torch.cat([motif_final, context_feat], dim=1))
        fused = gate_weights * motif_final + (1 - gate_weights) * context_feat
        return torch.sigmoid(self.output(fused)).squeeze()


# =============================================================================
# DATA
# =============================================================================
class PositionalMotifDataset(Dataset):
    def __init__(self, df, fg, variety_cols, tissue_cols, stage_cols,
                 context_info, hard_examples=None, fit_scaler=False, scaler=None):
        self.binned_data = df[fg['binned']].values.astype(np.float32)
        self.relpos_data = df[fg['relative_position']].values.astype(np.float32)
        self.firstpos_data = df[fg['first_position']].values.astype(np.float32)
        self.lastpos_data = df[fg['last_position']].values.astype(np.float32)
        self.gradient_data = df[fg['density_gradient']].values.astype(np.float32)
        self.count_data = df[fg['global_count']].values.astype(np.float32)
        self.cooc_data = df[fg['cooccurrence']].values.astype(np.float32)
        self.seq_stats = df[fg['seq_stats']].values.astype(np.float32)
        self.variety_data = df[variety_cols].values.astype(np.float32)
        self.tissue_data = df[tissue_cols].values.astype(np.float32)
        self.stage_data = df[stage_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.float32)

        self.variety_tissue = context_info['variety_tissue']
        self.tissue_stage = context_info['tissue_stage']
        self.variety_stage = context_info['variety_stage']
        self.variety_idx = context_info['variety_idx']
        self.tissue_idx = context_info['tissue_idx']
        self.stage_idx = context_info['stage_idx']

        self.is_hard = (hard_examples.astype(np.float32) if hard_examples is not None
                        else np.zeros(len(self.labels), dtype=np.float32))

        if fit_scaler or scaler is not None:
            scalable = np.hstack([self.binned_data, self.count_data, self.cooc_data, self.seq_stats])
            if fit_scaler:
                self.scaler = StandardScaler()
                scaled = self.scaler.fit_transform(scalable)
            else:
                self.scaler = scaler
                scaled = self.scaler.transform(scalable)
            nb, nc, nco = (self.binned_data.shape[1], self.count_data.shape[1], self.cooc_data.shape[1])
            self.binned_data = scaled[:, :nb].astype(np.float32)
            self.count_data = scaled[:, nb:nb + nc].astype(np.float32)
            self.cooc_data = scaled[:, nb + nc:nb + nc + nco].astype(np.float32)
            self.seq_stats = scaled[:, nb + nc + nco:].astype(np.float32)
        else:
            self.scaler = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'binned': torch.FloatTensor(self.binned_data[idx]),
            'relpos': torch.FloatTensor(self.relpos_data[idx]),
            'firstpos': torch.FloatTensor(self.firstpos_data[idx]),
            'lastpos': torch.FloatTensor(self.lastpos_data[idx]),
            'gradient': torch.FloatTensor(self.gradient_data[idx]),
            'count': torch.FloatTensor(self.count_data[idx]),
            'cooc': torch.FloatTensor(self.cooc_data[idx]),
            'seq_stats': torch.FloatTensor(self.seq_stats[idx]),
            'variety': torch.FloatTensor(self.variety_data[idx]),
            'tissue': torch.FloatTensor(self.tissue_data[idx]),
            'stage': torch.FloatTensor(self.stage_data[idx]),
            'label': torch.FloatTensor([self.labels[idx]]),
            'is_hard': torch.FloatTensor([self.is_hard[idx]]),
            'variety_tissue': torch.FloatTensor(self.variety_tissue[idx]),
            'tissue_stage': torch.FloatTensor(self.tissue_stage[idx]),
            'variety_stage': torch.FloatTensor(self.variety_stage[idx]),
            'variety_idx': torch.LongTensor([self.variety_idx[idx]]),
            'tissue_idx': torch.LongTensor([self.tissue_idx[idx]]),
            'stage_idx': torch.LongTensor([self.stage_idx[idx]]),
        }


def shared_context_interactions(df, vcols, tcols, scols):
    """Fixed full-product interaction one-hots (consistent across drought/salt)."""
    vi = df[vcols].values.argmax(1)
    ti = df[tcols].values.argmax(1)
    si = df[scols].values.argmax(1)
    nv, nt, ns = len(vcols), len(tcols), len(scols)
    n = len(df)
    vt = np.zeros((n, nv * nt), np.float32); vt[np.arange(n), vi * nt + ti] = 1
    ts = np.zeros((n, nt * ns), np.float32); ts[np.arange(n), ti * ns + si] = 1
    vs = np.zeros((n, nv * ns), np.float32); vs[np.arange(n), vi * ns + si] = 1
    return {'variety_tissue': vt, 'n_vt': nv * nt, 'tissue_stage': ts, 'n_ts': nt * ns,
            'variety_stage': vs, 'n_vs': nv * ns,
            'variety_idx': vi, 'tissue_idx': ti, 'stage_idx': si}


def identify_hard_examples(train_df, other_df, count_cols):
    tc = np.round(train_df[count_cols].values, 4)
    th = [hash(tuple(r)) for r in tc]
    conf = pd.Series(train_df['label'].values).groupby(th).nunique()
    hard = set(conf[conf > 1].index)
    train_hard = np.array([h in hard for h in th])
    oc = np.round(other_df[count_cols].values, 4)
    oh = [hash(tuple(r)) for r in oc]
    other_hard = np.array([h in hard for h in oh])
    return train_hard, other_hard


# =============================================================================
# TRAINING
# =============================================================================
def set_seed(seed):
    import random
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


def hard_weighted_bce(pred, target, is_hard, hw=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    return (bce * (1.0 + is_hard * (hw - 1.0))).mean()


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(**batch)
            preds.extend(np.atleast_1d(out.cpu().numpy()))
            labels.extend(np.atleast_1d(batch['label'].squeeze().cpu().numpy()))
    preds, labels = np.array(preds), np.array(labels)
    return roc_auc_score(labels, preds), preds, labels


def train_model(model, train_loader, dev_loader, device, lr, epochs, patience,
                weight_decay, seed, only_params=None, tag=""):
    set_seed(seed)
    params = only_params if only_params is not None else model.parameters()
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    best_auc, best_state, wait = 0, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            out = model(**batch)
            loss = hard_weighted_bce(out, batch['label'].squeeze(), batch['is_hard'].squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        dev_auc, _, _ = evaluate(model, dev_loader, device)
        scheduler.step(dev_auc)
        if dev_auc > best_auc:
            best_auc, best_state, wait = dev_auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if (epoch + 1) % 25 == 0:
            print(f"        {tag} epoch {epoch+1}/{epochs}  dev AUC {dev_auc:.4f}  (best {best_auc:.4f})")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return best_auc


def metrics_from_preds(y_true, y_prob):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    yb = (y_prob >= best_t).astype(int)
    return {
        'test_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, yb)),
        'precision': float(precision_score(y_true, yb)),
        'recall': float(recall_score(y_true, yb)),
        'f1': float(f1_score(y_true, yb)),
        'mcc': float(matthews_corrcoef(y_true, yb)),
        'optimal_threshold': float(best_t),
    }


# =============================================================================
# MAIN
# =============================================================================
def load_features(name):
    p = os.path.join(DATA_DIR, f"{name}_features.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}_features.csv"))


def make_loaders(df, fg, vcols, tcols, scols, scaler, fit, hard, batch, shuffle):
    ctx = shared_context_interactions(df, vcols, tcols, scols)
    ds = PositionalMotifDataset(df, fg, vcols, tcols, scols, ctx,
                                hard_examples=hard, fit_scaler=fit, scaler=scaler)
    ld = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)
    return ld, ds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("TRANSFER LEARNING — drought → salt")
    print("=" * 72)
    print(f"Device: {device} | seeds: {args.num_seeds} | "
          f"hidden {args.hidden_dim} dropout {args.dropout} batch {args.batch_size}")

    fg = json.load(open(os.path.join(DATA_DIR, "shared_feature_groups.json")))
    vcols, tcols, scols = fg['variety'], fg['tissue'], fg['stage']
    n_motifs = len(fg['global_count'])
    n_cooc = len(fg['cooccurrence'])
    count_cols = fg['global_count']

    dro = load_features("drought")
    sal = load_features("salt")
    print(f"drought {dro.shape} | salt {sal.shape} | n_motifs {n_motifs} n_cooc {n_cooc}")

    # ── splits ──
    dro_tr, dro_dv = train_test_split(dro, test_size=0.1, random_state=RANDOM_SEED, stratify=dro['label'])
    sal_tr, sal_tmp = train_test_split(sal, test_size=0.2, random_state=RANDOM_SEED, stratify=sal['label'])
    sal_dv, sal_te = train_test_split(sal_tmp, test_size=0.5, random_state=RANDOM_SEED, stratify=sal_tmp['label'])
    print(f"drought: train {len(dro_tr)} dev {len(dro_dv)} | "
          f"salt: train {len(sal_tr)} dev {len(sal_dv)} test {len(sal_te)}")

    # fixed model context dims (full product over shared vocab)
    nv, nt, ns = len(vcols), len(tcols), len(scols)
    n_vt, n_ts, n_vs = nv * nt, nt * ns, nv * ns

    # ── scalers + hard examples (per domain) ──
    dro_hard_tr, dro_hard_dv = identify_hard_examples(dro_tr, dro_dv, count_cols)
    sal_hard_tr, sal_hard_dv = identify_hard_examples(sal_tr, sal_dv, count_cols)
    _, sal_hard_te = identify_hard_examples(sal_tr, sal_te, count_cols)

    # drought loaders (fit scaler on drought train)
    dro_tr_ld, dro_tr_ds = make_loaders(dro_tr, fg, vcols, tcols, scols, None, True, dro_hard_tr, args.batch_size, True)
    dro_dv_ld, _ = make_loaders(dro_dv, fg, vcols, tcols, scols, dro_tr_ds.scaler, False, dro_hard_dv, args.batch_size, False)

    # salt loaders (fit scaler on salt train)
    sal_tr_ld, sal_tr_ds = make_loaders(sal_tr, fg, vcols, tcols, scols, None, True, sal_hard_tr, args.batch_size, True)
    sal_dv_ld, _ = make_loaders(sal_dv, fg, vcols, tcols, scols, sal_tr_ds.scaler, False, sal_hard_dv, args.batch_size, False)
    sal_te_ld, _ = make_loaders(sal_te, fg, vcols, tcols, scols, sal_tr_ds.scaler, False, sal_hard_te, args.batch_size, False)

    y_test = sal_te['label'].values.astype(int)

    def new_model(seed):
        set_seed(seed)
        return ContextEnhancedGatedFusionV5(
            n_motifs=n_motifs, n_bins=NUM_BINS, n_cooc=n_cooc,
            n_varieties=nv, n_tissues=nt, n_stages=ns, num_seq_stats=4,
            hidden_dim=args.hidden_dim, embed_dim=64, dropout=args.dropout,
            n_vt=n_vt, n_ts=n_ts, n_vs=n_vs).to(device)

    seeds = [42 + i * 1000 for i in range(args.num_seeds)]
    preds = {'scratch': [], 'transfer_full': [], 'transfer_frozen': []}
    pretrain_dev_aucs, salt_dev = {'scratch': [], 'transfer_full': [], 'transfer_frozen': []}, {}

    t0 = time.time()
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")

        # (A) scratch — salt only
        m = new_model(seed)
        a = train_model(m, sal_tr_ld, sal_dv_ld, device, args.scratch_lr,
                        args.finetune_epochs, args.finetune_patience, args.weight_decay,
                        seed, tag="scratch")
        _, p, _ = evaluate(m, sal_te_ld, device)
        preds['scratch'].append(p)
        print(f"    scratch        : salt-dev AUC {a:.4f}")

        # pretrain on drought (shared by B and C)
        mp = new_model(seed)
        pa = train_model(mp, dro_tr_ld, dro_dv_ld, device, args.pretrain_lr,
                        args.pretrain_epochs, args.pretrain_patience, args.weight_decay,
                        seed, tag="pretrain")
        pre_state = copy.deepcopy(mp.state_dict())
        torch.save(pre_state, os.path.join(CKPT_DIR, f"pretrained_seed{seed}.pt"))
        print(f"    pretrain(drought): drought-dev AUC {pa:.4f}")

        # (B) transfer_full — fine-tune all weights on salt
        m = new_model(seed); m.load_state_dict(pre_state)
        a = train_model(m, sal_tr_ld, sal_dv_ld, device, args.finetune_lr,
                        args.finetune_epochs, args.finetune_patience, args.weight_decay,
                        seed, tag="ft_full")
        _, p, _ = evaluate(m, sal_te_ld, device)
        preds['transfer_full'].append(p)
        print(f"    transfer_full  : salt-dev AUC {a:.4f}")

        # (C) transfer_frozen — freeze encoder, train output head only
        m = new_model(seed); m.load_state_dict(pre_state)
        for prm in m.encoder_parameters():
            prm.requires_grad = False
        a = train_model(m, sal_tr_ld, sal_dv_ld, device, args.finetune_lr,
                        args.finetune_epochs, args.finetune_patience, args.weight_decay,
                        seed, only_params=[p for p in m.parameters() if p.requires_grad],
                        tag="ft_frozen")
        _, p, _ = evaluate(m, sal_te_ld, device)
        preds['transfer_frozen'].append(p)
        print(f"    transfer_frozen: salt-dev AUC {a:.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    # ── ensemble (mean) + metrics ──
    summary = {}
    for cond in ['scratch', 'transfer_full', 'transfer_frozen']:
        ens = np.mean(np.array(preds[cond]), axis=0)
        m = metrics_from_preds(y_test, ens)
        summary[cond] = m
        pd.DataFrame({'true_label': y_test, 'predicted_prob': ens,
                      'predicted_label': (ens >= m['optimal_threshold']).astype(int)}
                     ).to_csv(os.path.join(RESULTS_DIR, f"tl_predictions_{cond}.csv"), index=False)

    rows = []
    print("\n" + "=" * 72)
    print("RESULTS  (salt test set, ensemble of %d seeds)" % args.num_seeds)
    print("=" * 72)
    hdr = f"{'condition':16} {'AUC':>7} {'PR-AUC':>7} {'F1':>7} {'MCC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
    print(hdr)
    for cond in ['scratch', 'transfer_full', 'transfer_frozen']:
        m = summary[cond]
        print(f"{cond:16} {m['test_auc']:7.4f} {m['pr_auc']:7.4f} {m['f1']:7.4f} "
              f"{m['mcc']:7.4f} {m['accuracy']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f}")
        rows.append({'condition': cond, **m})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, "tl_comparison.csv"), index=False, float_format="%.4f")
    json.dump(summary, open(os.path.join(RESULTS_DIR, "tl_comparison.json"), "w"), indent=2)

    delta = summary['transfer_full']['test_auc'] - summary['scratch']['test_auc']
    print(f"\nΔAUC (transfer_full − scratch): {delta:+.4f}")
    print(f"Saved results to {RESULTS_DIR}/")

    # ── bar chart ──
    metrics = ['test_auc', 'f1', 'mcc', 'accuracy']
    conds = ['scratch', 'transfer_full', 'transfer_frozen']
    x = np.arange(len(metrics)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, cond in enumerate(conds):
        ax.bar(x + (j - 1) * w, [summary[cond][mm] for mm in metrics], w, label=cond)
    ax.set_xticks(x); ax.set_xticklabels(['AUC', 'F1', 'MCC', 'Accuracy'])
    ax.set_ylabel('Score'); ax.set_title('Drought→Salt Transfer Learning (salt test)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, "tl_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved tl_comparison.png")


if __name__ == "__main__":
    main()
