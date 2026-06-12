#!/usr/bin/env python3
"""
Step 4: Train CARCM Classifier on Homology-Aware Split
=======================================================
Wrapper that patches DATA_DIR, RESULTS_BASE_DIR, CHECKPOINT_DIR
to point to the homology-aware split data, then runs the
standard rice_classifier_v7b.py training.
"""

import os, sys

BASE = "/home/saiful/CARCM/Homology aware gene level split"
OUTPUT_DIR = os.path.join(BASE, "output")
RESULTS_DIR = os.path.join(BASE, "models")
CHECKPOINT_DIR = os.path.join(BASE, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Set GPU before importing torch
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "5")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

# Patch the classifier module's paths before its __main__ code runs
sys.path.insert(0, "/home/saiful/CARCM")
import rice_classifier_v7b as clf_module

clf_module.DATA_DIR = OUTPUT_DIR
clf_module.RESULTS_BASE_DIR = RESULTS_DIR
clf_module.CHECKPOINT_DIR = CHECKPOINT_DIR

# Re-create dirs with patched paths
os.makedirs(clf_module.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(clf_module.RESULTS_BASE_DIR, exist_ok=True)

# Monkey-patch: homology split has 0 hard test examples (no overlap).
# Wrap roc_auc_score to handle empty arrays instead of crashing.
import numpy as np
from sklearn.metrics import roc_auc_score as _orig_roc_auc_score

def _safe_roc_auc_score(y_true, y_score, **kwargs):
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    return _orig_roc_auc_score(y_true, y_score, **kwargs)

clf_module.roc_auc_score = _safe_roc_auc_score

if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {GPU} → CUDA device 0")

    args = clf_module.parse_args()

    print("=" * 70)
    print("STEP 4: CARCM TRAINING — HOMOLOGY-AWARE SPLIT")
    print("=" * 70)
    print(f"  DATA_DIR:       {clf_module.DATA_DIR}")
    print(f"  RESULTS_DIR:    {clf_module.RESULTS_BASE_DIR}")
    print(f"  CHECKPOINT_DIR: {clf_module.CHECKPOINT_DIR}")
    print(f"  GPU:            {GPU}")
    print()

    config = {
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_seeds": args.num_seeds,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "run_name": args.run_name,
    }
    results = clf_module.run_experiment(config)
