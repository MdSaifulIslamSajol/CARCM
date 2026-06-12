"""
Shared utilities for the Diversity Scaling Study.

Implements:
- Context label recovery from one-hot columns.
- Curve A (diversity) and Curve B (sample-matched control) subsampling.
- Global context-interaction maps so that every subset shares the same
  model-input dimensions (critical for Curve B where training may include
  only one variety/tissue/stage).
- A lean train-and-evaluate function that reuses the same
  ContextEnhancedGatedFusionV5 architecture and loss as the main CARCM
  pipeline, but trains a single model per run (3 seeds per point).

Key design choice:
    The full training, dev, and test DataFrames are concatenated once to
    build the context-interaction lookup tables. These tables are then
    applied to every subsample so the model's n_vt/n_ts/n_vs never change
    regardless of which contexts are present in the training subset.
"""
from __future__ import annotations

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

# rice_classifier_v7b.py hard-codes CUDA_VISIBLE_DEVICES="5" at module import,
# which would clobber whatever the parent bash script has set. Preserve our
# launcher's choice by snapshotting CVD, importing, and restoring it BEFORE
# torch initializes CUDA (torch init is lazy, triggered only by first tensor
# allocation on a cuda device, so restoring here is effective).
_ORIG_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", None)

sys.path.insert(0, "/home/saiful/CARCM")
from rice_classifier_v7b import (  # noqa: E402
    ContextEnhancedGatedFusionV5,
    PositionalMotifDataset,
    identify_hard_examples,
    train_single_model,
    set_seed,
    NUM_BINS,
)

if _ORIG_CVD is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _ORIG_CVD
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


# =============================================================================
# CONTEXT RECOVERY AND SUBSAMPLING
# =============================================================================

PANEL_KEY = {"variety": "variety", "tissue": "tissue", "stage": "stage"}


def get_context_labels(df: pd.DataFrame, feature_groups: dict, panel: str) -> pd.Series:
    """Recover original categorical context labels from one-hot columns."""
    cols = feature_groups[PANEL_KEY[panel]]
    active = df[cols].idxmax(axis=1)
    prefix = cols[0].split("_", 1)[0] + "_"
    return active.str.replace(prefix, "", regex=False)


def subsample_curve_a(
    train_df: pd.DataFrame,
    context_labels: pd.Series,
    ordered_contexts: list,
    k: int,
    n_per: int,
    seed: int,
) -> pd.DataFrame:
    """Curve A (diversity): k contexts x n_per samples each.

    If a context has fewer than n_per samples, take all of them.
    """
    selected = ordered_contexts[:k]
    subsets = []
    for ctx in selected:
        mask = (context_labels == ctx).values
        pool = train_df[mask]
        take = min(len(pool), n_per)
        subsets.append(pool.sample(n=take, random_state=seed))
    return pd.concat(subsets, ignore_index=True).reset_index(drop=True)


def subsample_curve_b(
    train_df: pd.DataFrame,
    context_labels: pd.Series,
    dominant_context: str,
    k: int,
    n_per: int,
    seed: int,
) -> pd.DataFrame:
    """Curve B (sample-matched control): one dominant context x k*n_per total.

    If the dominant context has fewer samples than k*n_per, use everything.
    """
    mask = (context_labels == dominant_context).values
    pool = train_df[mask]
    target = k * n_per
    take = min(len(pool), target)
    return pool.sample(n=take, random_state=seed).reset_index(drop=True)


# =============================================================================
# GLOBAL CONTEXT-INTERACTION MAPS
# =============================================================================

def build_global_interactions(all_df: pd.DataFrame, feature_groups: dict) -> dict:
    """Pre-compute variety/tissue/stage interaction lookup tables on the FULL
    dataset so that every subsample yields identical tensor shapes."""
    variety_cols = feature_groups["variety"]
    tissue_cols = feature_groups["tissue"]
    stage_cols = feature_groups["stage"]

    n_v = len(variety_cols)
    n_t = len(tissue_cols)
    n_s = len(stage_cols)

    v_idx = all_df[variety_cols].values.argmax(axis=1)
    t_idx = all_df[tissue_cols].values.argmax(axis=1)
    s_idx = all_df[stage_cols].values.argmax(axis=1)

    vt_all = v_idx * n_t + t_idx
    ts_all = t_idx * n_s + s_idx
    vs_all = v_idx * n_s + s_idx

    vt_unique = np.unique(vt_all)
    ts_unique = np.unique(ts_all)
    vs_unique = np.unique(vs_all)

    return {
        "vt_map": {int(v): i for i, v in enumerate(vt_unique)},
        "ts_map": {int(v): i for i, v in enumerate(ts_unique)},
        "vs_map": {int(v): i for i, v in enumerate(vs_unique)},
        "n_vt": len(vt_unique),
        "n_ts": len(ts_unique),
        "n_vs": len(vs_unique),
        "n_varieties": n_v,
        "n_tissues": n_t,
        "n_stages": n_s,
        "variety_cols": variety_cols,
        "tissue_cols": tissue_cols,
        "stage_cols": stage_cols,
    }


def apply_global_interactions(df: pd.DataFrame, gm: dict) -> dict:
    """Produce the context_info dict expected by PositionalMotifDataset, using
    pre-computed global maps so shapes are fixed across subsets."""
    n_v = gm["n_varieties"]
    n_t = gm["n_tissues"]
    n_s = gm["n_stages"]

    v_idx = df[gm["variety_cols"]].values.argmax(axis=1)
    t_idx = df[gm["tissue_cols"]].values.argmax(axis=1)
    s_idx = df[gm["stage_cols"]].values.argmax(axis=1)

    vt_idx = v_idx * n_t + t_idx
    ts_idx = t_idx * n_s + s_idx
    vs_idx = v_idx * n_s + s_idx

    vt_mapped = np.array([gm["vt_map"][int(x)] for x in vt_idx])
    ts_mapped = np.array([gm["ts_map"][int(x)] for x in ts_idx])
    vs_mapped = np.array([gm["vs_map"][int(x)] for x in vs_idx])

    n = len(df)
    vt_oh = np.zeros((n, gm["n_vt"]), dtype=np.float32)
    vt_oh[np.arange(n), vt_mapped] = 1.0
    ts_oh = np.zeros((n, gm["n_ts"]), dtype=np.float32)
    ts_oh[np.arange(n), ts_mapped] = 1.0
    vs_oh = np.zeros((n, gm["n_vs"]), dtype=np.float32)
    vs_oh[np.arange(n), vs_mapped] = 1.0

    return {
        "variety_tissue": vt_oh,
        "tissue_stage": ts_oh,
        "variety_stage": vs_oh,
        "n_vt": gm["n_vt"],
        "n_ts": gm["n_ts"],
        "n_vs": gm["n_vs"],
        "variety_idx": v_idx.astype(np.int64),
        "tissue_idx": t_idx.astype(np.int64),
        "stage_idx": s_idx.astype(np.int64),
    }


# =============================================================================
# TRAIN + EVALUATE A SINGLE RUN
# =============================================================================

def train_and_eval_single(
    train_sub: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: dict,
    gm: dict,
    config: dict,
    seed: int,
    device: torch.device,
) -> dict:
    """Train one model on train_sub, early-stop on dev_df, evaluate on test_df.

    Returns a dict with all key metrics on the fixed test set.
    """
    set_seed(seed)

    variety_cols = feature_groups["variety"]
    tissue_cols = feature_groups["tissue"]
    stage_cols = feature_groups["stage"]
    count_cols = feature_groups["global_count"]

    # Global context-interaction tensors (stable shape across subsets)
    ctx_train = apply_global_interactions(train_sub, gm)
    ctx_dev = apply_global_interactions(dev_df, gm)
    ctx_test = apply_global_interactions(test_df, gm)

    # Hard-example flags (same logic as main pipeline)
    train_hard, test_hard, _ = identify_hard_examples(train_sub, test_df, count_cols)
    _, dev_hard, _ = identify_hard_examples(train_sub, dev_df, count_cols)

    # Datasets
    train_ds = PositionalMotifDataset(
        train_sub, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=ctx_train, hard_examples=train_hard, fit_scaler=True,
    )
    dev_ds = PositionalMotifDataset(
        dev_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=ctx_dev, hard_examples=dev_hard, scaler=train_ds.scaler,
    )
    test_ds = PositionalMotifDataset(
        test_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=ctx_test, hard_examples=test_hard, scaler=train_ds.scaler,
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Model
    model = ContextEnhancedGatedFusionV5(
        n_motifs=len(count_cols),
        n_bins=NUM_BINS,
        n_cooc=train_ds.n_cooc,
        n_varieties=len(variety_cols),
        n_tissues=len(tissue_cols),
        n_stages=len(stage_cols),
        hidden_dim=config["hidden_dim"],
        embed_dim=config.get("embed_dim", 64),
        dropout=config["dropout"],
        n_vt=gm["n_vt"],
        n_ts=gm["n_ts"],
        n_vs=gm["n_vs"],
    ).to(device)

    # Train
    model, best_dev_auc, _ = train_single_model(
        model, train_loader, dev_loader, device, config, seed
    )

    # Evaluate on fixed test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            labels = batch["label"].squeeze().cpu().numpy().flatten()
            output = model(**batch).cpu().numpy().flatten()
            all_preds.extend(output.tolist())
            all_labels.extend(labels.tolist())

    y = np.array(all_labels)
    p = np.array(all_preds)
    yhat = (p >= 0.5).astype(int)

    # Free model memory
    del model
    torch.cuda.empty_cache()

    return {
        "test_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "test_pr_auc": float(average_precision_score(y, p)),
        "test_f1": float(f1_score(y, yhat)),
        "test_mcc": float(matthews_corrcoef(y, yhat)),
        "test_acc": float(accuracy_score(y, yhat)),
        "test_bal_acc": float(balanced_accuracy_score(y, yhat)),
        "dev_auc": float(best_dev_auc),
        "n_train": int(len(train_sub)),
        "n_pos_train": int((train_sub["label"] == 1).sum()),
        "n_neg_train": int((train_sub["label"] == 0).sum()),
    }


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

def default_config() -> dict:
    """Training hyperparameters used for every scaling run.

    Matches the best CARCM config from run_v7c_experiments.sh except:
    - num_epochs and patience are reduced because training sets are small
      (1000-8000 samples) so convergence is fast; this keeps the 114-run
      budget manageable without changing optimizer, loss, or architecture.
    """
    return {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "hidden_dim": 128,
        "embed_dim": 64,
        "dropout": 0.4,
        "num_epochs": 80,
        "patience": 15,
    }
