#!/usr/bin/env python3
"""
Step 3: Feature Extraction using Homology-Aware Split
=====================================================
Reuses the positional motif pipeline from rice_positional_pipeline.py
but applies the homology-aware cluster-based train/dev/test split
instead of the random split.
"""

import os, sys, json
import numpy as np
import pandas as pd

BASE = "/home/saiful/CARCM/Homology aware gene level split"
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import the pipeline's feature extraction functions
sys.path.insert(0, "/home/saiful/CARCM")
from rice_positional_pipeline import (
    load_and_clean_data, create_positional_features,
    create_context_features, find_top_cooccurrence_pairs,
    MOTIFS, NUM_BINS, TOP_COOCCURRENCE_PAIRS, RANDOM_SEED, INPUT_FILE
)


def main():
    print("=" * 70)
    print("STEP 3: FEATURE EXTRACTION (HOMOLOGY-AWARE SPLIT)")
    print("=" * 70)

    # ── 1. Load the split assignment ─────────────────────────────────────────
    split_df = pd.read_csv(os.path.join(BASE, "samples_with_split.csv"))
    print(f"Loaded split assignment: {len(split_df)} samples")
    print(f"  Train: {(split_df['split']=='train').sum()}, "
          f"Dev: {(split_df['split']=='dev').sum()}, "
          f"Test: {(split_df['split']=='test').sum()}")

    # ── 2. Load original data and extract features ───────────────────────────
    print("\nLoading original data for feature extraction...")
    df = load_and_clean_data(INPUT_FILE)

    # Align the split assignment back to the original df
    # Both have promoter_sequence+Variety+Tissue+Stage+Gene Regulation
    # Use a merge to map the split column
    df["seq_upper"] = df["promoter_sequence"].str.upper().str.strip()
    split_df["seq_upper_s"] = split_df["promoter_sequence"].str.upper().str.strip()

    # Create a unique key for each sample
    df["merge_key"] = (df["seq_upper"] + "|" +
                       df["Variety"].astype(str) + "|" +
                       df["Tissue"].astype(str) + "|" +
                       df["Stage"].astype(str) + "|" +
                       df["Gene Regulation"].astype(str))

    split_df["merge_key"] = (split_df["seq_upper_s"] + "|" +
                             split_df["Variety"].astype(str) + "|" +
                             split_df["Tissue"].astype(str) + "|" +
                             split_df["Stage"].astype(str) + "|" +
                             split_df["Gene Regulation"].astype(str))

    split_lookup = dict(zip(split_df["merge_key"], split_df["split"]))
    df["split"] = df["merge_key"].map(split_lookup)

    n_matched = df["split"].notna().sum()
    print(f"  Matched {n_matched}/{len(df)} samples to split assignments")

    if df["split"].isna().sum() > 0:
        print(f"  Dropping {df['split'].isna().sum()} unmatched samples")
        df = df.dropna(subset=["split"]).copy()
    df = df.drop(columns=["seq_upper", "merge_key"])

    # ── 3. Extract features for all samples at once ──────────────────────────
    print("\nExtracting features...")
    sample_df = df.sample(min(5000, len(df)), random_state=RANDOM_SEED)
    top_cooc_pairs = find_top_cooccurrence_pairs(sample_df, TOP_COOCCURRENCE_PAIRS)

    with open(os.path.join(OUTPUT_DIR, "top_cooccurrence_pairs.txt"), "w") as f:
        for pair in top_cooc_pairs:
            f.write(pair + "\n")

    positional_df = create_positional_features(df, top_cooc_pairs)
    context_df, variety_cols, tissue_cols, stage_cols = create_context_features(df)

    final_df = pd.concat([positional_df, context_df], axis=1)
    final_df["label"] = df["label"].values
    final_df["split"] = df["split"].values

    print(f"\nFinal feature matrix: {final_df.shape}")

    # ── 4. Split and save ────────────────────────────────────────────────────
    train_df = final_df[final_df["split"] == "train"].drop(columns=["split"])
    dev_df = final_df[final_df["split"] == "dev"].drop(columns=["split"])
    test_df = final_df[final_df["split"] == "test"].drop(columns=["split"])

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%)")
    print(f"  Dev:   {len(dev_df)} ({len(dev_df)/len(final_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)")

    for sp in ["train", "dev", "test"]:
        sub = eval(f"{sp}_df")
        up = sub["label"].sum()
        print(f"  {sp:5s}: UP={int(up)}, DOWN={int(len(sub)-up)}")

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(OUTPUT_DIR, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    feature_names = [c for c in final_df.columns if c not in ("label", "split")]
    with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    binned_cols = [c for c in feature_names if "_bin" in c]
    relpos_cols = [c for c in feature_names if "_rel_pos" in c]
    firstpos_cols = [c for c in feature_names if "_first_pos" in c]
    lastpos_cols = [c for c in feature_names if "_last_pos" in c]
    gradient_cols = [c for c in feature_names if "_density_grad" in c]
    count_cols = [c for c in feature_names if c.endswith("_count")]
    cooc_cols = [c for c in feature_names if c.startswith("cooc_")]
    seq_stat_cols = ["length", "gc_content", "at_content", "gc_skew"]

    feature_groups = {
        "binned": binned_cols,
        "relative_position": relpos_cols,
        "first_position": firstpos_cols,
        "last_position": lastpos_cols,
        "density_gradient": gradient_cols,
        "global_count": count_cols,
        "cooccurrence": cooc_cols,
        "seq_stats": seq_stat_cols,
        "variety": variety_cols,
        "tissue": tissue_cols,
        "stage": stage_cols,
    }
    with open(os.path.join(OUTPUT_DIR, "feature_groups.json"), "w") as f:
        json.dump(feature_groups, f, indent=2)

    print(f"\n  Saved train/dev/test CSV + feature metadata → {OUTPUT_DIR}")
    print("  DONE.")


if __name__ == "__main__":
    main()
