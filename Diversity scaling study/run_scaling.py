#!/usr/bin/env python3
"""
Run a single panel (variety / tissue / stage) of the diversity scaling study.

For the chosen panel:
    Curve A - Diversity:   train on k contexts x N_per samples each (k = 1..K).
    Curve B - Control:     train on 1 dominant context x (k * N_per) samples.
Evaluate each trained model on the fixed 80/10/10 test set, with 3 seeds
per (k, curve) point. All per-run metrics are appended to a panel CSV.

Usage:
    CUDA_VISIBLE_DEVICES=2 python run_scaling.py --panel variety
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from scaling_utils import (  # noqa: E402
    get_context_labels,
    subsample_curve_a,
    subsample_curve_b,
    build_global_interactions,
    train_and_eval_single,
    default_config,
)


# =============================================================================
# PANEL CONFIGURATION
# =============================================================================
#
# For each panel we define:
#   k_values:          which k points to train (minimum 1, maximum = #contexts
#                      in TRAIN with >= n_per samples, constrained by dominant
#                      pool size for Curve B).
#   n_per:             per-context budget. User approved N=1000 for variety.
#                      Tissue uses 900 so Curve B tops out at 7200 samples
#                      which Seedling (7320) can fully cover without aliasing.
#                      Stage uses 1000 because Reproductive (20k+) dominates.
#   dominant_context:  single context used for Curve B.
#   ordered_contexts:  ordered by descending sample count in TRAIN
#                      (top-k used for Curve A).
#
# =============================================================================
PANELS = {
    "variety": {
        "n_per": 1000,
        "k_values": [1, 2, 3, 4, 5, 6],
        "dominant_context": "IR64_Cleaned",
        "ordered_contexts": [
            "IR64_Cleaned", "N22_Cleaned3", "Dhaggadeshi",
            "Nipponbare", "Moroberekan", "IR20",
        ],
    },
    "tissue": {
        "n_per": 900,
        "k_values": [1, 2, 3, 4, 5, 6, 7, 8],
        "dominant_context": "Seedling",
        "ordered_contexts": [
            "Seedling", "Root", "Panicle", "Leaf",
            "Flag Leaf", "Inflorescence", "Anther", "Pistil",
        ],
    },
    "stage": {
        "n_per": 1000,
        "k_values": [1, 2, 3, 4, 5],
        "dominant_context": "Reproductive Stage",
        "ordered_contexts": [
            "Reproductive Stage", "Vegetative Stage", "Seedling Stage",
            "Panicle Elongation Stage", "Tillering Stage",
        ],
    },
}

SEEDS = [42, 123, 2024]
DATA_DIR = Path("/home/saiful/CARCM/output")
RESULTS_DIR = THIS_DIR / "results"
LOG_DIR = THIS_DIR / "per_run_logs"


# =============================================================================

def run_panel(panel: str):
    cfg = PANELS[panel]
    run_cfg = default_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{panel}] device = {device} | visible GPUs = {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    print(f"[{panel}] Loading data ...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    with open(DATA_DIR / "feature_groups.json") as f:
        feature_groups = json.load(f)

    print(f"[{panel}] train={len(train_df)}  dev={len(dev_df)}  test={len(test_df)}")

    # Global interaction maps so all subsets share the same input shape
    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    gm = build_global_interactions(all_df, feature_groups)
    print(f"[{panel}] global interactions: n_vt={gm['n_vt']}, n_ts={gm['n_ts']}, n_vs={gm['n_vs']}")

    ctx_labels = get_context_labels(train_df, feature_groups, panel)
    print(f"[{panel}] train {panel} counts:\n{ctx_labels.value_counts().to_string()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"scaling_{panel}.csv"

    # Append mode: if file exists and a run is already recorded, skip it.
    done_keys = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done_keys = set(zip(prev["curve"], prev["k"], prev["seed"]))
        print(f"[{panel}] resuming: {len(done_keys)} runs already complete.")
    else:
        with open(out_csv, "w") as f:
            f.write(
                "panel,curve,k,seed,n_train,n_pos_train,n_neg_train,"
                "test_auc,test_pr_auc,test_f1,test_mcc,test_acc,test_bal_acc,"
                "dev_auc,runtime_sec\n"
            )

    total_runs = len(cfg["k_values"]) * 2 * len(SEEDS)
    completed = len(done_keys)
    panel_t0 = time.time()

    for k in cfg["k_values"]:
        for curve in ["A", "B"]:
            for seed in SEEDS:
                if (curve, k, seed) in done_keys:
                    continue
                tag = f"{panel}_curve{curve}_k{k}_seed{seed}"
                t0 = time.time()
                try:
                    if curve == "A":
                        train_sub = subsample_curve_a(
                            train_df, ctx_labels, cfg["ordered_contexts"],
                            k, cfg["n_per"], seed,
                        )
                    else:
                        train_sub = subsample_curve_b(
                            train_df, ctx_labels, cfg["dominant_context"],
                            k, cfg["n_per"], seed,
                        )

                    print(
                        f"[{panel}] >>> {tag}  n_train={len(train_sub)}  "
                        f"({completed+1}/{total_runs})"
                    )

                    res = train_and_eval_single(
                        train_sub, dev_df, test_df,
                        feature_groups, gm, run_cfg, seed, device,
                    )
                    runtime = time.time() - t0

                    row = (
                        f"{panel},{curve},{k},{seed},{res['n_train']},"
                        f"{res['n_pos_train']},{res['n_neg_train']},"
                        f"{res['test_auc']:.6f},{res['test_pr_auc']:.6f},"
                        f"{res['test_f1']:.6f},{res['test_mcc']:.6f},"
                        f"{res['test_acc']:.6f},{res['test_bal_acc']:.6f},"
                        f"{res['dev_auc']:.6f},{runtime:.1f}\n"
                    )
                    with open(out_csv, "a") as f:
                        f.write(row)

                    print(
                        f"[{panel}] <<< {tag}  AUC={res['test_auc']:.4f}  "
                        f"MCC={res['test_mcc']:.3f}  "
                        f"t={runtime:.1f}s"
                    )
                    completed += 1
                except Exception as e:
                    print(f"[{panel}] !!! {tag} FAILED: {e}")
                    traceback.print_exc()
                    with open(LOG_DIR / f"{tag}_ERROR.txt", "w") as f:
                        f.write(f"{tag}\n{e}\n{traceback.format_exc()}")

    elapsed = (time.time() - panel_t0) / 60
    print(f"[{panel}] DONE panel in {elapsed:.1f} min ({completed}/{total_runs} runs).")


# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=list(PANELS.keys()), required=True)
    args = ap.parse_args()
    run_panel(args.panel)
