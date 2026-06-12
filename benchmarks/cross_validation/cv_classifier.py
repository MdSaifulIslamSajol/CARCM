#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Validation Classifier — train one fold on a specified GPU.
================================================================
Thin wrapper around rice_classifier_v7b.py that:
  1. Accepts --fold and --gpu before any imports
  2. Locks CUDA to the requested GPU before torch initialises
  3. Monkey-patches the module-level paths so the classifier reads/writes
     from the correct fold directory

Usage:
    python cv_classifier.py --fold 0 --gpu 2 --learning_rate 1e-3 ...
"""

import os
import sys
import argparse

# ── Parse fold / gpu BEFORE importing torch ──────────────────────────────────
parser = argparse.ArgumentParser(description='CV Classifier — single fold')
parser.add_argument('--fold', type=int, required=True, help='Fold index (0-4)')
parser.add_argument('--gpu', type=str, required=True, help='GPU ID to use')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--num_seeds', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epochs', type=int, default=150)
parser.add_argument('--patience', type=int, default=25)
args = parser.parse_args()

# Lock the GPU *before* anything touches CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch  # noqa: E402  — must come after env-var is set
if torch.cuda.is_available():
    torch.cuda.init()  # force CUDA context so later env-var changes are no-ops

# ── Now import the classifier module ─────────────────────────────────────────
# rice_classifier_v7b.py line 24 sets CUDA_VISIBLE_DEVICES = "5",
# but our CUDA context is already bound to args.gpu above.
sys.path.insert(0, '/home/saiful/CARCM')
import rice_classifier_v7b as clf  # noqa: E402

# ── Override module-level paths for this fold ────────────────────────────────
CV_BASE = "/home/saiful/CARCM/benchmarks/cross_validation"

clf.DATA_DIR = os.path.join(CV_BASE, f"fold_{args.fold}", "output")
clf.RESULTS_BASE_DIR = os.path.join(CV_BASE, f"fold_{args.fold}", "models")
clf.CHECKPOINT_DIR = os.path.join(CV_BASE, f"fold_{args.fold}", "checkpoints")

os.makedirs(clf.RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(clf.CHECKPOINT_DIR, exist_ok=True)

# ── Run ──────────────────────────────────────────────────────────────────────
config = {
    'learning_rate': args.learning_rate,
    'hidden_dim': args.hidden_dim,
    'dropout': args.dropout,
    'num_seeds': args.num_seeds,
    'batch_size': args.batch_size,
    'weight_decay': args.weight_decay,
    'num_epochs': args.num_epochs,
    'patience': args.patience,
    'run_name': f'fold_{args.fold}',
}

print(f"{'=' * 70}")
print(f"CROSS-VALIDATION — Fold {args.fold} on GPU {args.gpu}")
print(f"{'=' * 70}")
print(f"  Data dir      : {clf.DATA_DIR}")
print(f"  Results dir   : {clf.RESULTS_BASE_DIR}")
print(f"  Checkpoint dir: {clf.CHECKPOINT_DIR}")
print(f"  CUDA device   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

results = clf.run_experiment(config)
