#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bin-count experiment (SALT) — regenerate features at NUM_BINS=N, then train the
winning hyperparameter config (combined_opt1: lr=1e-4, hidden=128, dropout=0.5,
batch=128, 10 seeds) on the SAME stratified 80/10/10 split (RANDOM_SEED=42), so
results are directly comparable to the 5-bin run (test AUC 0.6686).

Usage:
    python run_bins_experiment_salt.py --nbins 10 --gpu 0
    python run_bins_experiment_salt.py --nbins 20 --gpu 1
"""
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nbins', type=int, required=True)
parser.add_argument('--gpu', type=str, required=True)
args = parser.parse_args()

# Lock GPU before torch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CARCM_SALT_GPU"] = args.gpu
import torch  # noqa: E402
if torch.cuda.is_available():
    torch.cuda.init()

BASE = "/home/saiful/DeepRice_Salt/CARCM_Salt"
sys.path.insert(0, BASE)

NB = args.nbins
OUT_DIR = os.path.join(BASE, f"output_bins{NB}/")

# ── 1. Regenerate positional features at NUM_BINS=NB ─────────────────────────
import rice_positional_pipeline_salt as pl  # noqa: E402
pl.NUM_BINS = NB
pl.OUTPUT_DIR = OUT_DIR
print(f"\n########## FEATURE EXTRACTION @ NUM_BINS={NB} ##########\n")
pl.main()

# ── 2. Train the winning config on these features ────────────────────────────
import rice_classifier_v7b_salt as clf  # noqa: E402
clf.NUM_BINS = NB
clf.DATA_DIR = OUT_DIR
clf.RESULTS_BASE_DIR = os.path.join(BASE, "models_carcm_salt/")
clf.CHECKPOINT_DIR = os.path.join(BASE, "checkpoints/")
os.makedirs(clf.RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(clf.CHECKPOINT_DIR, exist_ok=True)

config = {
    'learning_rate': 1e-4,   # winning config = combined_opt1
    'hidden_dim': 128,
    'dropout': 0.5,
    'num_seeds': 10,
    'batch_size': 128,
    'weight_decay': 0.01,
    'num_epochs': 150,
    'patience': 25,
    'run_name': f'bins{NB}',
}

print(f"\n########## CLASSIFIER (winning config) @ NUM_BINS={NB} ##########\n")
clf.run_experiment(config)
