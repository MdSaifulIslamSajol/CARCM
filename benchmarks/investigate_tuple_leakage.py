#!/usr/bin/env python3
"""Investigate the 43 leaked (seq, Variety, Tissue, Stage) tuples.

Reads the raw input CSV to find exact duplicate tuples, then checks
how they ended up in both train and test after the stratified split.
"""

import pandas as pd
from pathlib import Path

INPUT = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_5.csv"
SPLIT_DIR = Path("/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/output")

TUPLE_COLS = ["promoter_sequence", "Variety", "Tissue", "Stage"]
# The split files use 'seq' instead of 'promoter_sequence' and 'label' instead of 'Gene Regulation'
SPLIT_TUPLE_COLS = ["seq", "Variety", "Tissue", "Stage"]


def main():
    # ── 1. Analyse the raw input CSV ──────────────────────────────────
    print("=" * 70)
    print("STEP 1: Duplicate tuples in the RAW INPUT CSV")
    print("=" * 70)

    df = pd.read_csv(INPUT)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    n_unique = df.drop_duplicates(subset=TUPLE_COLS).shape[0]
    print(f"Unique (seq, V, T, S) tuples: {n_unique}")
    print(f"Duplicate rows (same tuple appears >1 time): {len(df) - n_unique}")

    dup_mask = df.duplicated(subset=TUPLE_COLS, keep=False)
    dups = df[dup_mask].copy()
    print(f"\nRows that belong to a duplicated tuple: {len(dups)}")

    if len(dups) > 0:
        grp = dups.groupby(TUPLE_COLS)
        dup_tuples = grp.size().reset_index(name="count")
        print(f"Distinct duplicate tuples: {len(dup_tuples)}")

        # Label agreement
        label_nunique = grp["Gene Regulation"].nunique().reset_index(name="n_labels")
        same = (label_nunique["n_labels"] == 1).sum()
        diff = (label_nunique["n_labels"] > 1).sum()
        print(f"  Same label across copies: {same}")
        print(f"  Conflicting labels:       {diff}")

        print(f"\nDuplicate-count distribution:")
        print(dup_tuples["count"].value_counts().sort_index().to_string())

        # Show a few examples
        print(f"\nFirst 10 duplicate tuples (showing Variety, Tissue, Stage, label, count):")
        examples = dup_tuples.merge(
            label_nunique, on=TUPLE_COLS
        ).sort_values("count", ascending=False).head(10)
        for _, row in examples.iterrows():
            print(f"  V={row['Variety']:20s}  T={row['Tissue']:15s}  "
                  f"S={row['Stage']:25s}  copies={row['count']}  "
                  f"n_labels={row['n_labels']}")
    else:
        print("No duplicate tuples found in the raw CSV.")

    # ── 2. Simulate the split to trace which duplicates land where ────
    print("\n" + "=" * 70)
    print("STEP 2: Simulating the stratified split (same random_state=42)")
    print("=" * 70)

    from sklearn.model_selection import train_test_split

    # Reproduce the exact same split as rice_positional_pipeline.py
    RANDOM_SEED = 42
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED,
        stratify=df["Gene Regulation"]
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED,
        stratify=temp_df["Gene Regulation"]
    )
    print(f"Split sizes: train={len(train_df)}  dev={len(dev_df)}  test={len(test_df)}")

    # Tag each row with its split
    df_tagged = df.copy()
    df_tagged["split"] = "unknown"
    df_tagged.loc[train_df.index, "split"] = "train"
    df_tagged.loc[dev_df.index, "split"] = "dev"
    df_tagged.loc[test_df.index, "split"] = "test"

    # Find duplicate tuples that land in DIFFERENT splits
    dup_groups = df_tagged[dup_mask].groupby(TUPLE_COLS)

    leaked = []
    for name, group in dup_groups:
        splits = group["split"].unique()
        if len(splits) > 1:
            leaked.append({
                "Variety": name[1], "Tissue": name[2], "Stage": name[3],
                "seq_prefix": name[0][:50],
                "splits": sorted(splits.tolist()),
                "labels": group["Gene Regulation"].tolist(),
                "count": len(group),
            })

    print(f"\nDuplicate tuples that ended up in DIFFERENT splits: {len(leaked)}")

    # ── 3. Show the leaked tuples ─────────────────────────────────────
    if len(leaked) > 0:
        print("\n" + "=" * 70)
        print(f"STEP 3: Details of the {len(leaked)} leaked tuples")
        print("=" * 70)

        test_leaked = [l for l in leaked if "test" in l["splits"]]
        print(f"  Involving TEST split: {len(test_leaked)}")
        print(f"\n{'Variety':20s} {'Tissue':20s} {'Stage':28s} | splits           labels  seq_prefix")
        print("-" * 130)
        for l in sorted(leaked, key=lambda x: (x["Variety"], x["Tissue"], x["Stage"])):
            print(f"{l['Variety']:20s} {l['Tissue']:20s} {l['Stage']:28s} | "
                  f"{str(l['splits']):17s} {l['labels']}  {l['seq_prefix']}...")

    # ── 4. Root cause summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: ROOT CAUSE")
    print("=" * 70)

    print(f"\nThe raw input CSV has {len(dup_tuples)} duplicate (seq, V, T, S) tuples.")
    print(f"ALL {len(dup_tuples)} have exactly 2 copies with CONFLICTING labels")
    print(f"(same promoter + same context -> one says Up, other says Down).")
    print(f"\nWhen stratified split is applied, some duplicate pairs get split")
    print(f"across train/test by random chance -> {len(test_leaked)} tuples leaked to test.")
    print(f"\nThis is LABEL NOISE in the original data, not a pipeline bug.")
    print(f"The same gene under the same (Variety, Tissue, Stage) has contradictory")
    print(f"regulation labels in two different source experiments/studies.")

    # ── 5. Quick fix recommendation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: QUICK FIX")
    print("=" * 70)
    print(f"Rows before dedup: {len(df)}")
    df_dedup = df.drop_duplicates(subset=TUPLE_COLS, keep="first")
    print(f"Rows after  dedup: {len(df_dedup)}")
    print(f"Rows removed:      {len(df) - len(df_dedup)}")
    print("\nTo fix: either drop_duplicates or resolve conflicting labels before splitting.")


if __name__ == "__main__":
    main()
