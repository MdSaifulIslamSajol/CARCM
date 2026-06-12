#!/usr/bin/env python3
"""
Step 2: Stratified cluster-level train/dev/test split
======================================================
- Assign entire clusters to train (80%) or test (20%)
- Further split train into train (87.5%) + dev (12.5%) → overall 70/10/20
- Preserve variety/tissue/stage/label distributions as closely as possible
- Generate split statistics and distribution comparison plots
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

BASE = "/home/saiful/CARCM/Homology aware gene level split"
np.random.seed(42)


def main():
    print("=" * 70)
    print("STEP 2: STRATIFIED CLUSTER-LEVEL TRAIN/DEV/TEST SPLIT")
    print("=" * 70)

    df = pd.read_csv(os.path.join(BASE, "samples_with_clusters.csv"))
    print(f"Loaded {len(df)} samples with {df['cluster_id'].nunique()} clusters")

    # Build a cluster-level summary for stratified assignment
    cluster_info = df.groupby("cluster_id").agg(
        n_samples=("promoter_sequence", "size"),
        label_mean=("Gene Regulation", lambda x: (x == "Up Regulated").mean()),
        n_genes=("gene_id", "nunique"),
        majority_variety=("Variety", lambda x: x.mode()[0]),
        majority_tissue=("Tissue", lambda x: x.mode()[0]),
    ).reset_index()

    # Stratification key: majority variety + label tendency (above/below median)
    median_label = cluster_info["label_mean"].median()
    cluster_info["strat_key"] = (
        cluster_info["majority_variety"].astype(str) + "_" +
        np.where(cluster_info["label_mean"] >= median_label, "hi", "lo")
    )

    # Shuffle clusters, then perform stratified split
    clusters = cluster_info["cluster_id"].values.copy()
    np.random.shuffle(clusters)

    strat_key_map = dict(zip(cluster_info["cluster_id"], cluster_info["strat_key"]))

    # Group clusters by strat_key
    from collections import defaultdict
    strat_groups = defaultdict(list)
    for c in clusters:
        strat_groups[strat_key_map[c]].append(c)

    test_clusters = set()
    train_clusters_full = set()
    target_test_frac = 0.20

    for key, group_clusters in strat_groups.items():
        n_test = max(1, int(len(group_clusters) * target_test_frac))
        test_clusters.update(group_clusters[:n_test])
        train_clusters_full.update(group_clusters[n_test:])

    # Further split train into train + dev
    train_clusters_list = list(train_clusters_full)
    np.random.shuffle(train_clusters_list)
    n_dev = max(1, int(len(train_clusters_list) * 0.125))  # 12.5% of train → ~10% overall
    dev_clusters = set(train_clusters_list[:n_dev])
    train_clusters = set(train_clusters_list[n_dev:])

    df["split"] = "train"
    df.loc[df["cluster_id"].isin(dev_clusters), "split"] = "dev"
    df.loc[df["cluster_id"].isin(test_clusters), "split"] = "test"

    print(f"\nCluster-level split:")
    print(f"  Train clusters: {len(train_clusters)} ({len(train_clusters)/(len(train_clusters)+len(dev_clusters)+len(test_clusters))*100:.1f}%)")
    print(f"  Dev   clusters: {len(dev_clusters)} ({len(dev_clusters)/(len(train_clusters)+len(dev_clusters)+len(test_clusters))*100:.1f}%)")
    print(f"  Test  clusters: {len(test_clusters)} ({len(test_clusters)/(len(train_clusters)+len(dev_clusters)+len(test_clusters))*100:.1f}%)")

    print(f"\nSample-level split:")
    for sp in ["train", "dev", "test"]:
        sub = df[df["split"] == sp]
        n = len(sub)
        up = (sub["Gene Regulation"] == "Up Regulated").sum()
        print(f"  {sp:5s}: {n:6d} samples ({n/len(df)*100:.1f}%), "
              f"Up={up} ({up/n*100:.1f}%), Down={n-up} ({(n-up)/n*100:.1f}%)")

    # Verify: no gene overlap between splits
    train_genes = set(df[df["split"] == "train"]["gene_id"])
    dev_genes = set(df[df["split"] == "dev"]["gene_id"])
    test_genes = set(df[df["split"] == "test"]["gene_id"])

    overlap_train_test = train_genes & test_genes
    overlap_train_dev = train_genes & dev_genes
    overlap_dev_test = dev_genes & test_genes
    print(f"\nGene overlap verification:")
    print(f"  Train ∩ Test:  {len(overlap_train_test)} genes {'✓ NONE' if len(overlap_train_test)==0 else '✗ OVERLAP!'}")
    print(f"  Train ∩ Dev:   {len(overlap_train_dev)} genes {'✓ NONE' if len(overlap_train_dev)==0 else '✗ OVERLAP!'}")
    print(f"  Dev   ∩ Test:  {len(overlap_dev_test)} genes {'✓ NONE' if len(overlap_dev_test)==0 else '✗ OVERLAP!'}")

    # Distribution comparison
    print("\nDistribution comparison (train vs test):")
    for col in ["Variety", "Tissue", "Stage"]:
        train_dist = df[df["split"] == "train"][col].value_counts(normalize=True)
        test_dist = df[df["split"] == "test"][col].value_counts(normalize=True)
        all_vals = sorted(set(train_dist.index) | set(test_dist.index))
        max_diff = max(abs(train_dist.get(v, 0) - test_dist.get(v, 0)) for v in all_vals)
        print(f"  {col}: max absolute difference = {max_diff:.3f}")

    # Save split assignment
    df.to_csv(os.path.join(BASE, "samples_with_split.csv"), index=False)

    # Save cluster assignments as JSON
    split_info = {
        "train_clusters": sorted(int(c) for c in train_clusters),
        "dev_clusters": sorted(int(c) for c in dev_clusters),
        "test_clusters": sorted(int(c) for c in test_clusters),
        "n_train": int(len(df[df["split"] == "train"])),
        "n_dev": int(len(df[df["split"] == "dev"])),
        "n_test": int(len(df[df["split"] == "test"])),
    }
    with open(os.path.join(BASE, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # ── Visualization ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Homology-Aware Gene-Level Split — Distribution Analysis",
                 fontsize=14, fontweight="bold")

    # (a) Split sizes
    ax = axes[0, 0]
    split_counts = df["split"].value_counts()
    colors = {"train": "#4CAF50", "dev": "#FF9800", "test": "#F44336"}
    bars = ax.bar(split_counts.index, split_counts.values,
                  color=[colors[s] for s in split_counts.index])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{int(bar.get_height())}", ha="center", fontsize=10)
    ax.set_title("(a) Split Sizes", fontweight="bold")
    ax.set_ylabel("Number of Samples")

    # (b) Class balance per split
    ax = axes[0, 1]
    class_data = df.groupby(["split", "Gene Regulation"]).size().unstack(fill_value=0)
    class_data = class_data.div(class_data.sum(axis=1), axis=0)
    class_data[["Up Regulated", "Down Regulated"]].plot(kind="bar", stacked=True, ax=ax,
                                     color=["#FF5722", "#2196F3"])
    ax.set_title("(b) Class Balance per Split", fontweight="bold")
    ax.set_ylabel("Proportion")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Regulation")

    # (c) Variety distribution
    ax = axes[1, 0]
    for sp in ["train", "test"]:
        dist = df[df["split"] == sp]["Variety"].value_counts(normalize=True).sort_index()
        ax.bar(range(len(dist)), dist.values, alpha=0.6, label=sp, color=colors[sp])
    ax.set_title("(c) Variety Distribution", fontweight="bold")
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels(dist.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion")
    ax.legend()

    # (d) Cluster size distribution
    ax = axes[1, 1]
    cluster_sizes = df.groupby("cluster_id").size()
    ax.hist(cluster_sizes, bins=50, color="#9C27B0", edgecolor="white", alpha=0.8)
    ax.set_title(f"(d) Cluster Size Distribution (n={len(cluster_sizes)} clusters)",
                 fontweight="bold")
    ax.set_xlabel("Samples per Cluster")
    ax.set_ylabel("Count")
    ax.axvline(cluster_sizes.median(), color="red", linestyle="--",
               label=f"Median: {cluster_sizes.median():.0f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "split_distribution.png"), dpi=400, bbox_inches="tight")
    plt.savefig(os.path.join(BASE, "split_distribution.pdf"), bbox_inches="tight")
    plt.close()
    print("\nSaved split distribution figure.")

    # Save split statistics CSV
    stats = []
    for sp in ["train", "dev", "test"]:
        sub = df[df["split"] == sp]
        stats.append({
            "split": sp,
            "n_samples": len(sub),
            "n_genes": sub["gene_id"].nunique(),
            "n_clusters": sub["cluster_id"].nunique(),
            "pct_up": (sub["Gene Regulation"] == "Up Regulated").mean() * 100,
            "n_varieties": sub["Variety"].nunique(),
            "n_tissues": sub["Tissue"].nunique(),
            "n_stages": sub["Stage"].nunique(),
        })
    pd.DataFrame(stats).to_csv(os.path.join(BASE, "split_statistics.csv"), index=False)
    print("Saved split_statistics.csv")
    print("Done.")


if __name__ == "__main__":
    main()
