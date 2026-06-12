#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5-Fold Cross-Validation Pipeline — Feature Extraction & Stratified Splitting
=============================================================================
Extracts positional motif features for ALL data once, then creates 5 stratified
fold splits. Each fold directory gets its own train.csv, dev.csv, test.csv.

Output structure:
    cross_validation/
        fold_0/output/  {train.csv, dev.csv, test.csv, feature_groups.json, ...}
        fold_1/output/
        ...
        fold_4/output/
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/saiful/CARCM')
from rice_positional_pipeline import (
    load_and_clean_data, extract_positional_features,
    calculate_cooccurrence_features, find_top_cooccurrence_pairs,
    create_positional_features, create_context_features,
    INPUT_FILE, RANDOM_SEED, TOP_COOCCURRENCE_PAIRS,
)

N_FOLDS = 5
CV_BASE_DIR = "/home/saiful/CARCM/benchmarks/cross_validation"


def main():
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION — FEATURE EXTRACTION & SPLITTING")
    print("=" * 70)

    # ── Step 1: Load and clean data ──
    df = load_and_clean_data(INPUT_FILE)

    # ── Step 2: Find top co-occurrence pairs (from full dataset sample) ──
    sample_df = df.sample(min(5000, len(df)), random_state=RANDOM_SEED)
    top_cooc_pairs = find_top_cooccurrence_pairs(sample_df, TOP_COOCCURRENCE_PAIRS)

    # ── Step 3: Extract positional features for ALL data (done once) ──
    positional_df = create_positional_features(df, top_cooc_pairs)

    # ── Step 4: Create context features ──
    context_df, variety_cols, tissue_cols, stage_cols = create_context_features(df)

    # ── Step 5: Combine all features ──
    final_df = pd.concat([positional_df, context_df], axis=1)
    final_df['label'] = df['label'].values
    final_df = final_df.reset_index(drop=True)

    print(f"\nFull feature matrix: {final_df.shape}")

    # ── Build feature group metadata (required by classifier) ──
    binned_cols = [c for c in final_df.columns if '_bin' in c]
    relpos_cols = [c for c in final_df.columns if '_rel_pos' in c]
    firstpos_cols = [c for c in final_df.columns if '_first_pos' in c]
    lastpos_cols = [c for c in final_df.columns if '_last_pos' in c]
    gradient_cols = [c for c in final_df.columns if '_density_grad' in c]
    count_cols = [c for c in final_df.columns if c.endswith('_count')]
    cooc_cols = [c for c in final_df.columns if c.startswith('cooc_')]
    seq_stat_cols = ['length', 'gc_content', 'at_content', 'gc_skew']

    feature_groups = {
        'binned': binned_cols,
        'relative_position': relpos_cols,
        'first_position': firstpos_cols,
        'last_position': lastpos_cols,
        'density_gradient': gradient_cols,
        'global_count': count_cols,
        'cooccurrence': cooc_cols,
        'seq_stats': seq_stat_cols,
        'variety': variety_cols,
        'tissue': tissue_cols,
        'stage': stage_cols,
    }

    feature_names = [c for c in final_df.columns if c != 'label']

    # ── Step 6: StratifiedKFold splitting ──
    print(f"\n{'=' * 70}")
    print(f"CREATING {N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION SPLITS")
    print(f"{'=' * 70}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    labels = final_df['label'].values

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(final_df, labels)):
        fold_dir = os.path.join(CV_BASE_DIR, f"fold_{fold_idx}", "output")
        os.makedirs(fold_dir, exist_ok=True)

        train_val_df = final_df.iloc[train_val_idx]
        test_df = final_df.iloc[test_idx]

        # Split train_val into train (90%) and dev (10%)
        train_df, dev_df = train_test_split(
            train_val_df, test_size=0.1, random_state=RANDOM_SEED,
            stratify=train_val_df['label'],
        )

        print(f"\n  Fold {fold_idx}:")
        print(f"    Train: {len(train_df):>6d}  "
              f"(Up={int(train_df['label'].sum()):>5d}, "
              f"Down={len(train_df) - int(train_df['label'].sum()):>5d})")
        print(f"    Dev:   {len(dev_df):>6d}  "
              f"(Up={int(dev_df['label'].sum()):>5d}, "
              f"Down={len(dev_df) - int(dev_df['label'].sum()):>5d})")
        print(f"    Test:  {len(test_df):>6d}  "
              f"(Up={int(test_df['label'].sum()):>5d}, "
              f"Down={len(test_df) - int(test_df['label'].sum()):>5d})")

        # Save CSVs
        train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        dev_df.to_csv(os.path.join(fold_dir, 'dev.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

        # Save feature metadata (identical across folds but needed per-fold)
        with open(os.path.join(fold_dir, 'feature_names.txt'), 'w') as f:
            for name in feature_names:
                f.write(name + '\n')

        with open(os.path.join(fold_dir, 'feature_groups.json'), 'w') as f:
            json.dump(feature_groups, f, indent=2)

        with open(os.path.join(fold_dir, 'top_cooccurrence_pairs.txt'), 'w') as f:
            for pair in top_cooc_pairs:
                f.write(pair + '\n')

    print(f"\n{'=' * 70}")
    print(f"CV PIPELINE COMPLETE — {N_FOLDS} folds saved to {CV_BASE_DIR}/fold_*/output/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
