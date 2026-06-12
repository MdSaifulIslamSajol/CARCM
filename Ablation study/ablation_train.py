#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train one ablation configuration with N seeds and save per-seed test predictions
+ per-seed metrics.

For every seed:
  * Initialise an `AblationCARCM` with the configured switches
  * (For configuration #6, additionally permute the variety/tissue/stage
    one-hot columns independently in train+dev+test before building the
    interaction features)
  * Train on the same data/regime as the headline CARCM (LR=1e-3, hidden=128,
    dropout=0.4, batch=128) with early stopping on dev AUC
  * Save test predictions and per-seed metrics

Outputs (under <out_dir>/<config_name>/):
  predictions/seed_<seed>.npy        — test predictions (n,)
  test_labels.npy                    — shared test labels
  per_seed_metrics.csv               — auc/pr_auc/f1/mcc per seed
  history/seed_<seed>.json           — train/dev curves
  config.json                        — full run config

Usage:
  python ablation_train.py --config full      --gpu 2
  python ablation_train.py --config permuted_labels --gpu 3
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from ablation_model import (  # noqa: E402
    AblationCARCM, PositionalMotifDataset, NUM_BINS,
    create_context_interactions, identify_hard_examples,
    load_feature_groups, permute_context_columns, get_config,
)
from ablation_utils import metric_pack  # noqa: E402


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="ablation config name (see ABLATION_CONFIGS)")
    p.add_argument("--gpu", type=int, default=0)
    # Defaults match the v7c headline regime so Full CARCM reproduces ~0.852 AUC
    p.add_argument("--num_seeds", type=int, default=10)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--seed_step", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--data_dir", type=str,
                   default="/home/saiful/CARCM/output/")
    p.add_argument("--out_root", type=str,
                   default="/home/saiful/CARCM/Ablation study/results/")
    return p.parse_args()


# =============================================================================
# helpers
# =============================================================================

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def hard_weighted_bce_loss(pred, target, is_hard, hard_weight=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    weights = 1.0 + is_hard * (hard_weight - 1.0)
    return (bce * weights).mean()


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(**batch)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            preds.extend(out.cpu().numpy())
            labels.extend(batch["label"].squeeze().cpu().numpy())
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return preds, labels


def train_one(model, train_loader, dev_loader, device, cfg):
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )
    best_auc = -1.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "dev_auc": [], "lr": []}
    from sklearn.metrics import roc_auc_score

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            optimizer.zero_grad()
            out = model(**batch)
            loss = hard_weighted_bce_loss(
                out, batch["label"].squeeze(),
                batch["is_hard"].squeeze(), hard_weight=2.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        dev_preds, dev_labels = evaluate(model, dev_loader, device)
        dev_auc = float(roc_auc_score(dev_labels, dev_preds)) \
            if len(np.unique(dev_labels)) == 2 else float("nan")
        history["dev_auc"].append(dev_auc)
        scheduler.step(dev_auc if not np.isnan(dev_auc) else 0.0)

        if dev_auc > best_auc:
            best_auc = dev_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1:3d}  loss={avg_loss:.4f}  dev_auc={dev_auc:.4f}",
                  flush=True)
        if patience_counter >= cfg["patience"]:
            print(f"      early stop @ epoch {epoch+1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, history


# =============================================================================
# main
# =============================================================================

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    abl = get_config(args.config)
    out_dir = os.path.join(args.out_root, abl["name"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "predictions_dev"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "history"), exist_ok=True)

    cfg = {
        "learning_rate": args.learning_rate, "hidden_dim": args.hidden_dim,
        "dropout": args.dropout, "batch_size": args.batch_size,
        "weight_decay": args.weight_decay, "num_epochs": args.num_epochs,
        "patience": args.patience, "num_seeds": args.num_seeds,
        "ablation": abl, "data_dir": args.data_dir, "gpu": args.gpu,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("=" * 72, flush=True)
    print(f"ABLATION: id={abl['id']}  name={abl['name']}", flush=True)
    print(f"  label : {abl['label']}", flush=True)
    print(f"  switches: use_sequence={abl['use_sequence']}  "
          f"context_dims={abl['context_dims']}  "
          f"interactions={abl['use_interactions']}  "
          f"bias_replace={abl['bias_replace']}  "
          f"permute_context={abl['permute_context']}", flush=True)
    print(f"  GPU: {args.gpu}    seeds: {args.num_seeds}    "
          f"epochs: {args.num_epochs}    patience: {args.patience}", flush=True)
    print("=" * 72, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    # ---------------- load and (optionally) permute data ----------------
    print("loading data...", flush=True)
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    feature_groups = load_feature_groups(args.data_dir)

    variety_cols = [c for c in train_df.columns if c.startswith("Variety_")]
    tissue_cols = [c for c in train_df.columns if c.startswith("Tissue_")]
    stage_cols = [c for c in train_df.columns if c.startswith("Stage_")]
    print(f"  splits: train={len(train_df)}  dev={len(dev_df)}  test={len(test_df)}",
          flush=True)
    print(f"  contexts: variety={len(variety_cols)}  "
          f"tissue={len(tissue_cols)}  stage={len(stage_cols)}", flush=True)

    if abl["permute_context"]:
        print("PERMUTING context labels independently in train/dev/test (config #6)",
              flush=True)
        train_df = permute_context_columns(
            train_df, variety_cols, tissue_cols, stage_cols, seed=2026
        )
        dev_df = permute_context_columns(
            dev_df, variety_cols, tissue_cols, stage_cols, seed=2027
        )
        test_df = permute_context_columns(
            test_df, variety_cols, tissue_cols, stage_cols, seed=2028
        )

    # Fit a SHARED interaction vocabulary on the union of all splits so that
    # train/dev/test produce column-compatible interaction one-hots even when
    # context labels are permuted (configuration #6).
    union_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    union_ctx = create_context_interactions(union_df, variety_cols, tissue_cols, stage_cols)
    shared_vocab = union_ctx["vocab"]
    train_ctx = create_context_interactions(train_df, variety_cols, tissue_cols, stage_cols, vocab=shared_vocab)
    dev_ctx = create_context_interactions(dev_df, variety_cols, tissue_cols, stage_cols, vocab=shared_vocab)
    test_ctx = create_context_interactions(test_df, variety_cols, tissue_cols, stage_cols, vocab=shared_vocab)
    print(f"  shared interaction vocab: n_vt={train_ctx['n_vt']}  "
          f"n_ts={train_ctx['n_ts']}  n_vs={train_ctx['n_vs']}", flush=True)

    count_cols = feature_groups["global_count"]
    train_hard, test_hard, _ = identify_hard_examples(train_df, test_df, count_cols)
    _, dev_hard, _ = identify_hard_examples(train_df, dev_df, count_cols)

    train_ds = PositionalMotifDataset(
        train_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=train_ctx, hard_examples=train_hard, fit_scaler=True,
    )
    dev_ds = PositionalMotifDataset(
        dev_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=dev_ctx, hard_examples=dev_hard, scaler=train_ds.scaler,
    )
    test_ds = PositionalMotifDataset(
        test_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=test_ctx, hard_examples=test_hard, scaler=train_ds.scaler,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    n_motifs = 54
    n_bins = NUM_BINS
    n_cooc = len(feature_groups["cooccurrence"])
    n_varieties, n_tissues, n_stages = len(variety_cols), len(tissue_cols), len(stage_cols)

    # ---------------- save labels (reused by aggregator) ----------------
    test_labels = test_ds.labels.astype(np.int64)
    dev_labels = dev_ds.labels.astype(np.int64)
    np.save(os.path.join(out_dir, "test_labels.npy"), test_labels)
    np.save(os.path.join(out_dir, "dev_labels.npy"), dev_labels)

    # ---------------- per-seed training ----------------
    seeds = [args.seed_base + i * args.seed_step for i in range(cfg["num_seeds"])]
    rows = []
    t0 = time.time()
    for sidx, seed in enumerate(seeds):
        print(f"\n--- seed {sidx+1}/{len(seeds)}  ({seed}) ---", flush=True)
        set_seed(seed)
        model = AblationCARCM(
            n_motifs=n_motifs, n_bins=n_bins, n_cooc=n_cooc,
            n_varieties=n_varieties, n_tissues=n_tissues, n_stages=n_stages,
            num_seq_stats=4, hidden_dim=cfg["hidden_dim"],
            embed_dim=64, dropout=cfg["dropout"],
            n_vt=train_ctx["n_vt"], n_ts=train_ctx["n_ts"], n_vs=train_ctx["n_vs"],
            use_sequence=abl["use_sequence"],
            context_dims=abl["context_dims"],
            use_interactions=abl["use_interactions"],
            bias_replace=abl["bias_replace"],
            feature_mask=abl.get("feature_mask"),
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}", flush=True)

        model, dev_auc, history = train_one(model, train_loader, dev_loader, device, cfg)

        test_preds, _ = evaluate(model, test_loader, device)
        dev_preds, _ = evaluate(model, dev_loader, device)
        np.save(os.path.join(out_dir, "predictions", f"seed_{seed}.npy"), test_preds)
        np.save(os.path.join(out_dir, "predictions_dev", f"seed_{seed}.npy"), dev_preds)
        with open(os.path.join(out_dir, "history", f"seed_{seed}.json"), "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.items()}, f)

        m = metric_pack(test_labels, test_preds)
        m.update({"seed": int(seed), "dev_auc": float(dev_auc), "n_params": int(n_params)})
        rows.append(m)
        print(f"  [seed {seed}] test_auc={m['auc']:.4f}  pr_auc={m['pr_auc']:.4f}  "
              f"f1={m['f1']:.4f}  mcc={m['mcc']:.4f}  thr={m['threshold']:.2f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_seed_metrics.csv"), index=False)

    elapsed = (time.time() - t0) / 60.0
    print("\n" + "=" * 72, flush=True)
    print(f"DONE  config={abl['name']}  seeds={cfg['num_seeds']}  "
          f"elapsed={elapsed:.1f} min", flush=True)
    print(f"per-seed AUC: mean={df['auc'].mean():.4f}  std={df['auc'].std(ddof=1):.4f}",
          flush=True)
    print(f"results in: {out_dir}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
