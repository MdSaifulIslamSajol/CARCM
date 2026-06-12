#!/usr/bin/env python3
"""
Step 1: Match promoter sequences to RAP IDs + CD-HIT-EST clustering
=====================================================================
1. Read the FASTA file to build a sequence→RAP ID lookup
2. Match each sample in _6.csv to its RAP ID
3. Write unique promoters as FASTA for clustering
4. Run CD-HIT-EST at 80% identity
5. Parse cluster output
"""

import os, sys, subprocess, hashlib
import numpy as np
import pandas as pd

BASE = "/home/saiful/CARCM/Homology aware gene level split"
INPUT_CSV = "/home/saiful/CARCM/data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
FASTA_FILE = "/home/saiful/DeepRice/IRGSP-1.0_1kb-upstream_2025-03-19.fasta"

os.makedirs(BASE, exist_ok=True)


def read_fasta(path):
    """Read FASTA → dict of {header_id: sequence}."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(current_seq).upper()
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(current_seq).upper()
    return seqs


def main():
    print("=" * 70)
    print("STEP 1: MATCH RAP IDs + CLUSTER PROMOTERS")
    print("=" * 70)

    # ── 1. Load FASTA ────────────────────────────────────────────────────────
    print("\n[1/5] Loading reference FASTA...")
    fasta_seqs = read_fasta(FASTA_FILE)
    print(f"  Loaded {len(fasta_seqs)} sequences from FASTA")

    # Build sequence→RAP ID lookup (uppercase, strip for matching)
    seq_to_rap = {}
    for rap_id, seq in fasta_seqs.items():
        seq_upper = seq.upper().strip()
        seq_to_rap[seq_upper] = rap_id

    print(f"  Built sequence→RAP_ID lookup ({len(seq_to_rap)} unique sequences)")

    # ── 2. Load CSV and match ────────────────────────────────────────────────
    print("\n[2/5] Loading sample CSV and matching RAP IDs...")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["promoter_sequence"])
    df = df[df["promoter_sequence"].astype(str).str.len() >= 100].copy()
    print(f"  Samples after cleaning: {len(df)}")

    df["seq_upper"] = df["promoter_sequence"].str.upper().str.strip()

    matched = 0
    unmatched = 0
    rap_ids = []
    for seq in df["seq_upper"]:
        if seq in seq_to_rap:
            rap_ids.append(seq_to_rap[seq])
            matched += 1
        else:
            rap_ids.append(None)
            unmatched += 1

    df["rap_id"] = rap_ids

    print(f"  Matched: {matched} ({matched/len(df)*100:.1f}%)")
    print(f"  Unmatched: {unmatched} ({unmatched/len(df)*100:.1f}%)")

    if unmatched > 0:
        print(f"  Dropping {unmatched} unmatched samples...")
        df = df.dropna(subset=["rap_id"]).copy()

    # Convert transcript-style IDs to gene-level (Os01t0100100-01 → Os01g0100100)
    df["gene_id"] = df["rap_id"].str.split("-").str[0].str.replace("t", "g", n=1)
    n_genes = df["gene_id"].nunique()
    print(f"  Unique genes: {n_genes}")
    print(f"  Samples per gene (mean): {len(df)/n_genes:.1f}")

    # ── 3. Write unique promoters FASTA ──────────────────────────────────────
    print("\n[3/5] Writing unique promoter FASTA...")
    unique_genes = df.drop_duplicates(subset=["gene_id"])[["gene_id", "seq_upper", "rap_id"]]
    fasta_path = os.path.join(BASE, "unique_promoters.fa")
    with open(fasta_path, "w") as f:
        for _, row in unique_genes.iterrows():
            f.write(f">{row['gene_id']}\n{row['seq_upper']}\n")
    print(f"  Wrote {len(unique_genes)} unique promoters to {fasta_path}")

    # ── 4. Run CD-HIT-EST ────────────────────────────────────────────────────
    print("\n[4/5] Running CD-HIT-EST (80% identity, word size 5)...")
    cluster_out = os.path.join(BASE, "clusters")
    cmd = [
        "cd-hit-est",
        "-i", fasta_path,
        "-o", cluster_out,
        "-c", "0.80",
        "-n", "5",
        "-M", "16000",
        "-T", "8",
        "-d", "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print("  STDERR:", result.stderr[-500:])
        sys.exit(1)
    print("  CD-HIT-EST completed successfully.")

    # ── 5. Parse cluster output ──────────────────────────────────────────────
    print("\n[5/5] Parsing cluster assignments...")
    clstr_file = cluster_out + ".clstr"
    gene_to_cluster = {}
    current_cluster = None

    with open(clstr_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[-1])
            elif line:
                gene_id = line.split(">")[1].split("...")[0]
                gene_to_cluster[gene_id] = current_cluster

    n_clusters = len(set(gene_to_cluster.values()))
    print(f"  Total clusters: {n_clusters}")
    print(f"  Genes assigned: {len(gene_to_cluster)}")

    cluster_sizes = pd.Series(gene_to_cluster.values()).value_counts()
    print(f"  Cluster size: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"mean={cluster_sizes.mean():.1f}, median={cluster_sizes.median():.0f}")
    print(f"  Singleton clusters: {(cluster_sizes == 1).sum()} "
          f"({(cluster_sizes == 1).sum()/n_clusters*100:.1f}%)")

    # Map cluster to samples
    df["cluster_id"] = df["gene_id"].map(gene_to_cluster)
    unmapped = df["cluster_id"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} samples have no cluster assignment (dropping)")
        df = df.dropna(subset=["cluster_id"]).copy()
    df["cluster_id"] = df["cluster_id"].astype(int)

    # Save annotated data
    save_cols = ["promoter_sequence", "Variety", "Tissue", "Stage",
                 "Gene Regulation", "rap_id", "gene_id", "cluster_id"]
    out_path = os.path.join(BASE, "samples_with_clusters.csv")
    df[save_cols].to_csv(out_path, index=False)
    print(f"\n  Saved annotated data: {out_path} ({len(df)} samples)")

    # Summary stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total samples:    {len(df)}")
    print(f"  Unique genes:     {df['gene_id'].nunique()}")
    print(f"  Clusters (80%id): {n_clusters}")
    print(f"  Varieties:        {df['Variety'].nunique()}")
    print(f"  Tissues:          {df['Tissue'].nunique()}")
    print(f"  Stages:           {df['Stage'].nunique()}")


if __name__ == "__main__":
    main()
