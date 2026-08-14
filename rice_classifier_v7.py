#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rice Promoter Classifier V7 - Stacking Ensemble
=================================================
Uses V5's proven architecture with advanced ensemble techniques:
1. First Level: Train 10 diverse base models (V5 architecture)
2. Second Level: Meta-learner trained on base model predictions
3. Rank averaging for final ensemble
4. No k-mers (they hurt performance in V6b)

@author: saiful
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import copy

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================
import sys
# sys.stdout = open("/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/console_output_classifier_v7x.txt", "w")

# DATA_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/output/"
# OUTPUT_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/models_v7/"


sys.stdout = open("/home/saiful/CARCM/console_output_classifier_v7.txt", "w")

DATA_DIR = "/home/saiful/CARCM/output/"
OUTPUT_DIR = "/home/saiful/CARCM/models_carcm/"

# Training config (same as V5)
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 150
PATIENCE = 25
WEIGHT_DECAY = 0.01

# Model config (same as V5)
HIDDEN_DIM = 128
CONTEXT_EMBED_DIM = 64
DROPOUT = 0.4

# Stacking ensemble config
BASE_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]  # 10 base models
NUM_FOLDS = 5  # For out-of-fold predictions

MOTIF_NAMES = [
    'ABRE', 'ABRE_core', 'CE1', 'CE3', 'DRE_CRT', 'DRE_full', 'LTRE',
    'MYB_core', 'MYB1AT', 'MYB2', 'MYBCORE', 'MYC', 'WBOX', 'WBOX_core',
    'GT1', 'GT1_core', 'HSE', 'HSE_core', 'GCC_box', 'ERE', 'GBOX', 'IBOX',
    'CAAT_box', 'STRE', 'LTRE_like', 'JASE', 'TBOX', 'NAC_core', 'NACRS',
    'ACGT_core', 'ABRE_like', 'AuxRE', 'AuxRE_rev', 'ROOTMOTIF', 'AS1',
    'TATA_box', 'CACTFTPPCA1', 'ICE_box', 'CBF_DRE', 'ACGT', 'CACG', 'CATG',
    'CGTG', 'GCGT', 'AACA', 'CAAC', 'AAAG', 'GATA', 'TATA', 'CAAT', 'GCCG',
    'CGCG', 'TGAC', 'GTAC'
]

NUM_BINS = 5


# =============================================================================
# DATA PREPARATION (Same as V5)
# =============================================================================

def load_feature_groups(data_dir):
    with open(os.path.join(data_dir, 'feature_groups.json'), 'r') as f:
        return json.load(f)


def create_context_interactions(df, variety_cols, tissue_cols, stage_cols):
    variety_idx = df[variety_cols].values.argmax(axis=1)
    tissue_idx = df[tissue_cols].values.argmax(axis=1)
    stage_idx = df[stage_cols].values.argmax(axis=1)
    
    n_tissues = len(tissue_cols)
    n_stages = len(stage_cols)
    
    vt_idx = variety_idx * n_tissues + tissue_idx
    vt_unique = np.unique(vt_idx)
    vt_map = {v: i for i, v in enumerate(vt_unique)}
    vt_mapped = np.array([vt_map[v] for v in vt_idx])
    vt_onehot = np.zeros((len(df), len(vt_unique)), dtype=np.float32)
    vt_onehot[np.arange(len(df)), vt_mapped] = 1
    
    ts_idx = tissue_idx * n_stages + stage_idx
    ts_unique = np.unique(ts_idx)
    ts_map = {v: i for i, v in enumerate(ts_unique)}
    ts_mapped = np.array([ts_map[v] for v in ts_idx])
    ts_onehot = np.zeros((len(df), len(ts_unique)), dtype=np.float32)
    ts_onehot[np.arange(len(df)), ts_mapped] = 1
    
    vs_idx = variety_idx * n_stages + stage_idx
    vs_unique = np.unique(vs_idx)
    vs_map = {v: i for i, v in enumerate(vs_unique)}
    vs_mapped = np.array([vs_map[v] for v in vs_idx])
    vs_onehot = np.zeros((len(df), len(vs_unique)), dtype=np.float32)
    vs_onehot[np.arange(len(df)), vs_mapped] = 1
    
    return {
        'variety_tissue': vt_onehot, 'n_vt': len(vt_unique),
        'tissue_stage': ts_onehot, 'n_ts': len(ts_unique),
        'variety_stage': vs_onehot, 'n_vs': len(vs_unique),
        'variety_idx': variety_idx,
        'tissue_idx': tissue_idx,
        'stage_idx': stage_idx
    }


def identify_hard_examples(train_df, test_df, count_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_counts = np.round(train_df[count_cols].values, 4)
    train_hashes = [hash(tuple(row)) for row in train_counts]
    train_df['motif_hash'] = train_hashes
    
    conflicting = train_df.groupby('motif_hash')['label'].nunique()
    hard_promoters = set(conflicting[conflicting > 1].index)
    
    train_hard = train_df['motif_hash'].isin(hard_promoters).values
    
    test_counts = np.round(test_df[count_cols].values, 4)
    test_hashes = [hash(tuple(row)) for row in test_counts]
    test_df['motif_hash'] = test_hashes
    test_hard = test_df['motif_hash'].isin(hard_promoters).values
    
    return train_hard, test_hard, hard_promoters


class PositionalMotifDataset(Dataset):
    """Same as V5 dataset."""
    def __init__(self, df, feature_groups, variety_cols, tissue_cols, stage_cols,
                 context_info=None, hard_examples=None,
                 fit_scaler=False, scaler=None):
        
        binned_cols = feature_groups['binned']
        relpos_cols = feature_groups['relative_position']
        firstpos_cols = feature_groups['first_position']
        lastpos_cols = feature_groups['last_position']
        gradient_cols = feature_groups['density_gradient']
        count_cols = feature_groups['global_count']
        cooc_cols = feature_groups['cooccurrence']
        seq_stat_cols = feature_groups['seq_stats']
        
        self.binned_data = df[binned_cols].values.astype(np.float32)
        self.relpos_data = df[relpos_cols].values.astype(np.float32)
        self.firstpos_data = df[firstpos_cols].values.astype(np.float32)
        self.lastpos_data = df[lastpos_cols].values.astype(np.float32)
        self.gradient_data = df[gradient_cols].values.astype(np.float32)
        self.count_data = df[count_cols].values.astype(np.float32)
        self.cooc_data = df[cooc_cols].values.astype(np.float32)
        self.seq_stats = df[seq_stat_cols].values.astype(np.float32)
        
        self.variety_data = df[variety_cols].values.astype(np.float32)
        self.tissue_data = df[tissue_cols].values.astype(np.float32)
        self.stage_data = df[stage_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.float32)
        
        if context_info is not None:
            self.variety_tissue = context_info['variety_tissue']
            self.tissue_stage = context_info['tissue_stage']
            self.variety_stage = context_info['variety_stage']
            self.variety_idx = context_info['variety_idx']
            self.tissue_idx = context_info['tissue_idx']
            self.stage_idx = context_info['stage_idx']
        else:
            self.variety_tissue = None
        
        if hard_examples is not None:
            self.is_hard = hard_examples.astype(np.float32)
        else:
            self.is_hard = np.zeros(len(self.labels), dtype=np.float32)
        
        if fit_scaler:
            self.scaler = StandardScaler()
            scalable = np.hstack([self.binned_data, self.count_data, 
                                  self.cooc_data, self.seq_stats])
            scalable_scaled = self.scaler.fit_transform(scalable)
            
            n_binned = self.binned_data.shape[1]
            n_count = self.count_data.shape[1]
            n_cooc = self.cooc_data.shape[1]
            
            self.binned_data = scalable_scaled[:, :n_binned].astype(np.float32)
            self.count_data = scalable_scaled[:, n_binned:n_binned+n_count].astype(np.float32)
            self.cooc_data = scalable_scaled[:, n_binned+n_count:n_binned+n_count+n_cooc].astype(np.float32)
            self.seq_stats = scalable_scaled[:, n_binned+n_count+n_cooc:].astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            scalable = np.hstack([self.binned_data, self.count_data, 
                                  self.cooc_data, self.seq_stats])
            scalable_scaled = self.scaler.transform(scalable)
            
            n_binned = self.binned_data.shape[1]
            n_count = self.count_data.shape[1]
            n_cooc = self.cooc_data.shape[1]
            
            self.binned_data = scalable_scaled[:, :n_binned].astype(np.float32)
            self.count_data = scalable_scaled[:, n_binned:n_binned+n_count].astype(np.float32)
            self.cooc_data = scalable_scaled[:, n_binned+n_count:n_binned+n_count+n_cooc].astype(np.float32)
            self.seq_stats = scalable_scaled[:, n_binned+n_count+n_cooc:].astype(np.float32)
        else:
            self.scaler = None
        
        self.n_binned = self.binned_data.shape[1]
        self.n_relpos = self.relpos_data.shape[1]
        self.n_cooc = self.cooc_data.shape[1]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
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
            'is_hard': torch.FloatTensor([self.is_hard[idx]])
        }
        
        if self.variety_tissue is not None:
            item['variety_tissue'] = torch.FloatTensor(self.variety_tissue[idx])
            item['tissue_stage'] = torch.FloatTensor(self.tissue_stage[idx])
            item['variety_stage'] = torch.FloatTensor(self.variety_stage[idx])
            item['variety_idx'] = torch.LongTensor([self.variety_idx[idx]])
            item['tissue_idx'] = torch.LongTensor([self.tissue_idx[idx]])
            item['stage_idx'] = torch.LongTensor([self.stage_idx[idx]])
        
        return item


# =============================================================================
# MODEL - V5's Exact Architecture
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bin_attention = nn.Sequential(
            nn.Linear(n_bins, n_bins),
            nn.Softmax(dim=-1)
        )
        
        self.pos_fc = nn.Sequential(
            nn.Linear(n_motifs * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, binned, relpos, firstpos, lastpos, gradient):
        batch_size = binned.shape[0]
        
        binned_3d = binned.view(batch_size, self.n_motifs, self.n_bins)
        bin_weights = self.bin_attention(binned_3d)
        binned_weighted = binned_3d * bin_weights
        
        bin_features = self.bin_conv(binned_weighted)
        bin_features = bin_features.mean(dim=2)
        
        pos_concat = torch.cat([relpos, firstpos, lastpos, gradient], dim=1)
        pos_features = self.pos_fc(pos_concat)
        
        combined = torch.cat([bin_features, pos_features], dim=1)
        return self.combine(combined)


class ContextEnhancedGatedFusionV5(nn.Module):
    """V5's exact architecture."""
    def __init__(self, n_motifs, n_bins, n_cooc, n_varieties, n_tissues, n_stages,
                 num_seq_stats=4, hidden_dim=128, embed_dim=64, dropout=0.4,
                 n_vt=0, n_ts=0, n_vs=0):
        super().__init__()
        
        self.n_motifs = n_motifs
        self.n_bins = n_bins
        
        self.context_embed = ContextEmbeddingModule(
            n_varieties, n_tissues, n_stages, embed_dim
        )
        
        self.positional_proc = PositionalMotifProcessor(
            n_motifs, n_bins, hidden_dim, dropout
        )
        
        self.cooc_fc = nn.Sequential(
            nn.Linear(n_cooc, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.count_embed = nn.Linear(1, 32)
        self.lstm = nn.LSTM(32, hidden_dim // 2, num_layers=2, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.count_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        context_input_dim = embed_dim * 3 + n_vt + n_ts + n_vs + num_seq_stats
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        motif_feature_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        self.context_query = nn.Linear(hidden_dim, hidden_dim)
        self.motif_key = nn.Linear(motif_feature_dim, hidden_dim)
        self.motif_value = nn.Linear(motif_feature_dim, hidden_dim)
        self.attn_scale = hidden_dim ** 0.5
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, binned, relpos, firstpos, lastpos, gradient, count, cooc,
                seq_stats, variety, tissue, stage,
                variety_idx=None, tissue_idx=None, stage_idx=None,
                variety_tissue=None, tissue_stage=None, variety_stage=None, **kwargs):
        
        variety_emb, tissue_emb, stage_emb = self.context_embed(
            variety_idx, tissue_idx, stage_idx
        )
        
        pos_feat = self.positional_proc(binned, relpos, firstpos, lastpos, gradient)
        cooc_feat = self.cooc_fc(cooc)
        
        count_3d = self.count_embed(count.unsqueeze(-1))
        _, (h_n, _) = self.lstm(count_3d)
        count_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        count_feat = torch.relu(self.count_proj(count_feat))
        
        motif_combined = torch.cat([pos_feat, cooc_feat, count_feat], dim=1)
        
        context_parts = [variety_emb, tissue_emb, stage_emb, seq_stats]
        if variety_tissue is not None:
            context_parts.extend([variety_tissue, tissue_stage, variety_stage])
        context_vec = torch.cat(context_parts, dim=1)
        context_feat = self.context_fc(context_vec)
        
        query = self.context_query(context_feat)
        key = self.motif_key(motif_combined)
        value = self.motif_value(motif_combined)
        
        attn_score = (query * key).sum(dim=-1, keepdim=True) / self.attn_scale
        attn_weight = torch.sigmoid(attn_score)
        context_aware_motif = attn_weight * value
        
        motif_final = self.dropout(pos_feat) + context_aware_motif
        
        combined = torch.cat([motif_final, context_feat], dim=1)
        gate_weights = self.gate(combined)
        fused = gate_weights * motif_final + (1 - gate_weights) * context_feat
        
        return torch.sigmoid(self.output(fused)).squeeze()


# =============================================================================
# TRAINING
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # See rice_classifier_v7b.set_seed: without these, repeated runs of the same
    # configuration drift by roughly 0.002 test AUC.
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def hard_weighted_bce_loss(pred, target, is_hard, hard_weight=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    weights = 1.0 + is_hard * (hard_weight - 1.0)
    return (bce * weights).mean()


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_hard = []
    
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            outputs = model(**batch)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch['label'].squeeze().cpu().numpy())
            all_hard.extend(batch['is_hard'].squeeze().cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    hard = np.array(all_hard).astype(bool)
    
    auc_score = roc_auc_score(labels, preds)
    pred_binary = (preds > 0.5).astype(int)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary)
    
    easy_mask = ~hard
    results = {
        'auc': auc_score, 'acc': acc, 'f1': f1,
        'easy_auc': roc_auc_score(labels[easy_mask], preds[easy_mask]) if easy_mask.sum() > 10 else 0,
        'hard_auc': roc_auc_score(labels[hard], preds[hard]) if hard.sum() > 10 else 0,
        'easy_n': int(easy_mask.sum()),
        'hard_n': int(hard.sum()),
        'preds': preds,
        'labels': labels,
        'hard_mask': hard
    }
    
    return results


def get_predictions(model, loader, device):
    """Get predictions without computing metrics."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            outputs = model(**batch)
            all_preds.extend(outputs.cpu().numpy())
    
    return np.array(all_preds)


def find_optimal_threshold(preds, labels):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.2, 0.8, 0.01):
        pred_binary = (preds > thresh).astype(int)
        f1 = f1_score(labels, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def train_single_model(train_loader, dev_loader, model, device, seed,
                       num_epochs=150, patience=25, verbose=True):
    set_seed(seed)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_auc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = hard_weighted_bce_loss(outputs, batch['label'].squeeze(), batch['is_hard'].squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        dev_results = evaluate_model(model, dev_loader, device)
        
        if dev_results['auc'] > best_auc:
            best_auc = dev_results['auc']
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss/len(train_loader):.4f} | "
                  f"Dev AUC: {dev_results['auc']:.4f} | "
                  f"Hard: {dev_results['hard_auc']:.4f} | "
                  f"Best: {best_auc:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_auc


def rank_average(predictions_list):
    """Rank average multiple predictions."""
    n_samples = len(predictions_list[0])
    n_models = len(predictions_list)
    
    # Convert to ranks
    ranks = np.zeros((n_models, n_samples))
    for i, preds in enumerate(predictions_list):
        ranks[i] = rankdata(preds)
    
    # Average ranks and normalize
    avg_ranks = ranks.mean(axis=0)
    return avg_ranks / n_samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("RICE PROMOTER CLASSIFIER V7 - STACKING ENSEMBLE")
    print("10 Base Models + Meta-Learner + Rank Averaging")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    feature_groups = load_feature_groups(DATA_DIR)
    
    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    variety_cols = feature_groups['variety']
    tissue_cols = feature_groups['tissue']
    stage_cols = feature_groups['stage']
    count_cols = feature_groups['global_count']
    
    # Create context interactions
    train_context = create_context_interactions(train_df, variety_cols, tissue_cols, stage_cols)
    dev_context = create_context_interactions(dev_df, variety_cols, tissue_cols, stage_cols)
    test_context = create_context_interactions(test_df, variety_cols, tissue_cols, stage_cols)
    
    # Identify hard examples
    train_hard, _, _ = identify_hard_examples(train_df, train_df, count_cols)
    _, dev_hard, _ = identify_hard_examples(train_df, dev_df, count_cols)
    _, test_hard, _ = identify_hard_examples(train_df, test_df, count_cols)
    
    print(f"  Train hard: {train_hard.sum()} ({100*train_hard.mean():.1f}%)")
    
    # Create datasets
    train_dataset = PositionalMotifDataset(
        train_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=train_context, hard_examples=train_hard, fit_scaler=True
    )
    dev_dataset = PositionalMotifDataset(
        dev_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=dev_context, hard_examples=dev_hard, scaler=train_dataset.scaler
    )
    test_dataset = PositionalMotifDataset(
        test_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=test_context, hard_examples=test_hard, scaler=train_dataset.scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ==========================================================================
    # STAGE 1: Train Base Models
    # ==========================================================================
    print("\n" + "="*70)
    print(f"STAGE 1: TRAINING {len(BASE_SEEDS)} BASE MODELS")
    print("="*70)
    
    n_motifs = len(MOTIF_NAMES)
    n_cooc = len(feature_groups['cooccurrence'])
    
    base_models = []
    base_dev_aucs = []
    base_dev_preds = []
    base_test_preds = []
    
    for i, seed in enumerate(BASE_SEEDS):
        print(f"\n[{i+1}/{len(BASE_SEEDS)}] Training model with seed {seed}")
        
        model = ContextEnhancedGatedFusionV5(
            n_motifs=n_motifs,
            n_bins=NUM_BINS,
            n_cooc=n_cooc,
            n_varieties=len(variety_cols),
            n_tissues=len(tissue_cols),
            n_stages=len(stage_cols),
            num_seq_stats=len(feature_groups['seq_stats']),
            hidden_dim=HIDDEN_DIM,
            embed_dim=CONTEXT_EMBED_DIM,
            dropout=DROPOUT,
            n_vt=train_context['n_vt'],
            n_ts=train_context['n_ts'],
            n_vs=train_context['n_vs']
        ).to(device)
        
        if i == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {num_params:,}")
        
        model, best_auc = train_single_model(
            train_loader, dev_loader, model, device, seed,
            num_epochs=NUM_EPOCHS, patience=PATIENCE, verbose=(i < 3)
        )
        
        base_models.append(model)
        base_dev_aucs.append(best_auc)
        
        # Get predictions
        dev_preds = get_predictions(model, dev_loader, device)
        test_preds = get_predictions(model, test_loader, device)
        
        base_dev_preds.append(dev_preds)
        base_test_preds.append(test_preds)
        
        print(f"  Seed {seed}: Dev AUC = {best_auc:.4f}")
    
    # ==========================================================================
    # STAGE 2: Create Ensemble Predictions
    # ==========================================================================
    print("\n" + "="*70)
    print("STAGE 2: ENSEMBLE METHODS")
    print("="*70)
    
    dev_labels = dev_df['label'].values
    test_labels = test_df['label'].values
    test_hard_mask = test_hard.astype(bool)
    
    # Method 1: Simple Average
    simple_avg_dev = np.mean(base_dev_preds, axis=0)
    simple_avg_test = np.mean(base_test_preds, axis=0)
    simple_avg_auc = roc_auc_score(test_labels, simple_avg_test)
    simple_avg_hard = roc_auc_score(test_labels[test_hard_mask], simple_avg_test[test_hard_mask])
    print(f"\n1. Simple Average:")
    print(f"   Test AUC: {simple_avg_auc:.4f} | Hard AUC: {simple_avg_hard:.4f}")
    
    # Method 2: Weighted Average (by dev AUC)
    weights = np.array(base_dev_aucs)
    weights = weights / weights.sum()
    weighted_avg_dev = np.average(base_dev_preds, axis=0, weights=weights)
    weighted_avg_test = np.average(base_test_preds, axis=0, weights=weights)
    weighted_avg_auc = roc_auc_score(test_labels, weighted_avg_test)
    weighted_avg_hard = roc_auc_score(test_labels[test_hard_mask], weighted_avg_test[test_hard_mask])
    print(f"\n2. Weighted Average (by dev AUC):")
    print(f"   Test AUC: {weighted_avg_auc:.4f} | Hard AUC: {weighted_avg_hard:.4f}")
    
    # Method 3: Rank Average
    rank_avg_dev = rank_average(base_dev_preds)
    rank_avg_test = rank_average(base_test_preds)
    rank_avg_auc = roc_auc_score(test_labels, rank_avg_test)
    rank_avg_hard = roc_auc_score(test_labels[test_hard_mask], rank_avg_test[test_hard_mask])
    print(f"\n3. Rank Average:")
    print(f"   Test AUC: {rank_avg_auc:.4f} | Hard AUC: {rank_avg_hard:.4f}")
    
    # Method 4: Meta-Learner (Logistic Regression on base predictions)
    print(f"\n4. Training Meta-Learner (Logistic Regression)...")
    
    # Stack base predictions as features for meta-learner
    meta_train_features = np.column_stack(base_dev_preds)
    meta_test_features = np.column_stack(base_test_preds)
    
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_train_features, dev_labels)

    meta_test_preds = meta_model.predict_proba(meta_test_features)[:, 1]
    # The meta-learner is fitted on dev, so its dev score is computed out-of-fold
    # to keep the selection below fair.
    meta_dev_preds = cross_val_predict(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        meta_train_features, dev_labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method='predict_proba')[:, 1]
    meta_auc = roc_auc_score(test_labels, meta_test_preds)
    meta_hard = roc_auc_score(test_labels[test_hard_mask], meta_test_preds[test_hard_mask])
    print(f"   Test AUC: {meta_auc:.4f} | Hard AUC: {meta_hard:.4f}")

    # Method 5: Blended (Weighted avg + Rank avg)
    blended_dev = 0.5 * weighted_avg_dev + 0.5 * rank_avg_dev
    blended_test = 0.5 * weighted_avg_test + 0.5 * rank_avg_test
    blended_auc = roc_auc_score(test_labels, blended_test)
    blended_hard = roc_auc_score(test_labels[test_hard_mask], blended_test[test_hard_mask])
    print(f"\n5. Blended (Weighted + Rank):")
    print(f"   Test AUC: {blended_auc:.4f} | Hard AUC: {blended_hard:.4f}")

    # Select best ensemble method -- on DEV AUC, never on test.
    ensemble_results = {
        'simple_avg': (simple_avg_auc, simple_avg_hard, simple_avg_test, simple_avg_dev),
        'weighted_avg': (weighted_avg_auc, weighted_avg_hard, weighted_avg_test, weighted_avg_dev),
        'rank_avg': (rank_avg_auc, rank_avg_hard, rank_avg_test, rank_avg_dev),
        'meta_learner': (meta_auc, meta_hard, meta_test_preds, meta_dev_preds),
        'blended': (blended_auc, blended_hard, blended_test, blended_dev)
    }

    ensemble_dev_aucs = {k: roc_auc_score(dev_labels, v[3])
                         for k, v in ensemble_results.items()}
    best_name = max(ensemble_dev_aucs, key=ensemble_dev_aucs.get)
    best_auc, best_hard_auc, best_preds, best_dev_preds = ensemble_results[best_name]

    print(f"\n✓ Best Method (selected on dev AUC): {best_name}")
    print(f"  Dev AUC: {ensemble_dev_aucs[best_name]:.4f} | "
          f"Test AUC: {best_auc:.4f} | Hard AUC: {best_hard_auc:.4f}")
    
    # ==========================================================================
    # FINAL RESULTS
    # ==========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    easy_mask = ~test_hard_mask
    easy_auc = roc_auc_score(test_labels[easy_mask], best_preds[easy_mask])
    
    # Threshold tuned on the development split, then applied to test.
    opt_thresh, _ = find_optimal_threshold(best_dev_preds, dev_labels)
    opt_f1 = f1_score(test_labels, (best_preds > opt_thresh).astype(int))
    pred_opt = (best_preds > opt_thresh).astype(int)
    prec, rec, _, _ = precision_recall_fscore_support(test_labels, pred_opt, average='binary')
    
    print(f"\nBest Ensemble ({best_name}):")
    print(f"  Test AUC:  {best_auc:.4f}")
    print(f"  Hard AUC:  {best_hard_auc:.4f}")
    print(f"  Easy AUC:  {easy_auc:.4f}")
    print(f"  Optimal Threshold: {opt_thresh:.2f}")
    print(f"  F1 (optimal): {opt_f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_labels, pred_opt).ravel()
    print(f"\nConfusion Matrix (threshold={opt_thresh:.2f}):")
    print(f"  TN={tn:5d}  FP={fp:5d}")
    print(f"  FN={fn:5d}  TP={tp:5d}")
    
    # ==========================================================================
    # VERSION COMPARISON
    # ==========================================================================
    print("\n" + "="*70)
    print("VERSION COMPARISON")
    print("="*70)
    
    print(f"""
    Version    │  Test AUC  │  Hard AUC  │  vs V5
    ───────────┼────────────┼────────────┼──────────
    V1         │   0.7819   │   0.8403   │  baseline
    V3         │   0.8117   │   0.8932   │  +3.81%
    V5         │   0.8285   │   0.9128   │  +5.96%
    V7         │   {best_auc:.4f}   │   {best_hard_auc:.4f}   │  {(best_auc-0.8285)/0.8285*100:+.2f}%
    """)
    
    # ==========================================================================
    # SAVE
    # ==========================================================================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_summary = {
        'n_base_models': len(BASE_SEEDS),
        'base_seeds': BASE_SEEDS,
        'base_dev_aucs': base_dev_aucs,
        'ensemble_methods': {name: {'dev_auc': float(ensemble_dev_aucs[name]),
                                    'test_auc': float(auc), 'hard_auc': float(hard)}
                             for name, (auc, hard, _, _) in ensemble_results.items()},
        'best_method': best_name,
        'best_dev_auc': float(ensemble_dev_aucs[best_name]),
        'best_test_auc': float(best_auc),
        'best_hard_auc': float(best_hard_auc),
        'best_easy_auc': float(easy_auc),
        'best_f1': float(opt_f1),
        'optimal_threshold': float(opt_thresh),
        'precision': float(prec),
        'recall': float(rec),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save base models
    for model, seed in zip(base_models, BASE_SEEDS):
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'base_model_seed{seed}.pt'))
    
    # Save meta model
    import joblib
    joblib.dump(meta_model, os.path.join(OUTPUT_DIR, 'meta_model.pkl'))
    
    # Plot version comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    versions = ['V1', 'V3', 'V5', 'V7']
    test_aucs = [0.7819, 0.8117, 0.8285, best_auc]
    hard_aucs = [0.8403, 0.8932, 0.9128, best_hard_auc]
    
    x = np.arange(len(versions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_aucs, width, label='Test AUC', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, hard_aucs, width, label='Hard AUC', color='coral', alpha=0.8)
    
    ax.axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='Target 0.85')
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='Target 0.90')
    
    ax.set_ylabel('AUC')
    ax.set_title('Performance Evolution')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                   f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'version_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot ensemble comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(ensemble_results.keys())
    aucs = [ensemble_results[m][0] for m in methods]
    
    colors = ['coral' if m == best_name else 'steelblue' for m in methods]
    bars = ax.barh(methods, aucs, color=colors, alpha=0.8)
    
    ax.axvline(x=0.8285, color='green', linestyle='--', linewidth=2, label='V5 baseline')
    ax.set_xlabel('Test AUC')
    ax.set_title('Ensemble Methods Comparison')
    ax.legend()
    ax.set_xlim(0.78, 0.86)
    
    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.002, bar.get_y() + bar.get_height()/2,
               f'{auc:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll results saved to {OUTPUT_DIR}")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("V7 STACKING ENSEMBLE RESULTS")
    print("="*70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║               V7 STACKING ENSEMBLE RESULTS                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Best Method         │  {best_name:20}                ║
║  Base Models         │  {len(BASE_SEEDS)}                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Test AUC            │  {best_auc:.4f}                              ║
║  Hard Example AUC    │  {best_hard_auc:.4f}                              ║
║  Easy Example AUC    │  {easy_auc:.4f}                              ║
║  F1 Score            │  {opt_f1:.4f} (threshold={opt_thresh:.2f})           ║
╠══════════════════════════════════════════════════════════════════╣
║  Improvement vs V5   │  {(best_auc-0.8285)/0.8285*100:+.2f}%                             ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    if best_auc >= 0.85:
        print("\n🎉 TARGET ACHIEVED! Test AUC ≥ 0.85!")
    elif best_auc >= 0.83:
        print(f"\n📈 Close to target! Gap: {0.85 - best_auc:.4f}")
    
    print("\n" + "="*70)
    print("V7 COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
