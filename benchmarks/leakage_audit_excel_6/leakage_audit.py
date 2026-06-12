#!/usr/bin/env python3
"""Sequence-identity leakage audit via CD-HIT-EST at 80% identity.
Adapted for RiceMetaSys_Drought_merged_all_excels_6.csv (conflicting labels removed).
"""
import os
import sys
import subprocess
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

INPUT_CSV = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
OUT_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/benchmarks/leakage_audit_excel_6"
FASTA_ALL = os.path.join(OUT_DIR, "all_sequences.fasta")
FASTA_UNIQ = os.path.join(OUT_DIR, "unique_sequences.fasta")
CLSTR_PREFIX = os.path.join(OUT_DIR, "cdhit80")
LOG = os.path.join(OUT_DIR, "leakage_audit.log")
RANDOM_SEED = 42
IDENTITY = 0.80

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg, fh):
    print(msg); fh.write(msg + "\n")


def reproduce_splits():
    df = pd.read_csv(INPUT_CSV)
    df["label"] = (df["Gene Regulation"] == "Up Regulated").astype(int)
    df = df.dropna(subset=["promoter_sequence"]).reset_index(drop=True)
    df["seq_upper"] = df["promoter_sequence"].str.upper()

    train, temp = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    dev, test = train_test_split(
        temp, test_size=0.5, random_state=RANDOM_SEED, stratify=temp["label"]
    )
    train = train.assign(split="TRAIN").reset_index(drop=True)
    dev = dev.assign(split="DEV").reset_index(drop=True)
    test = test.assign(split="TEST").reset_index(drop=True)
    return pd.concat([train, dev, test], ignore_index=True)


def write_fasta(df, path, id_col, seq_col="seq_upper"):
    with open(path, "w") as f:
        for _, r in df.iterrows():
            f.write(f">{r[id_col]}\n{r[seq_col]}\n")


def run_cdhit(in_fa, out_prefix):
    exe = "/home/saiful/anaconda3/envs/dnabert/bin/cd-hit-est"
    cmd = [
        exe, "-i", in_fa, "-o", out_prefix,
        "-c", str(IDENTITY), "-n", "5", "-d", "0", "-M", "0", "-T", "0",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_clstr(path):
    clusters, current = [], None
    with open(path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                if current is not None:
                    clusters.append(current)
                current = []
            else:
                parts = line.strip().split(">")
                if len(parts) > 1:
                    name = parts[1].split("...")[0]
                    current.append(name)
        if current is not None:
            clusters.append(current)
    return clusters


def main():
    fh = open(LOG, "w")
    log("=" * 72, fh)
    log("SEQUENCE-IDENTITY LEAKAGE AUDIT  (CD-HIT-EST @ 80% identity)", fh)
    log("INPUT: RiceMetaSys_Drought_merged_all_excels_6.csv", fh)
    log("(conflicting-label rows removed)", fh)
    log("=" * 72, fh)

    log("\n[1] Reproducing train/dev/test split (random_state=42)", fh)
    full = reproduce_splits()
    split_counts = full["split"].value_counts().to_dict()
    log(f"    Split sizes: {split_counts}", fh)
    log(f"    Total samples: {len(full)}", fh)

    uniq = full.drop_duplicates(subset="seq_upper").reset_index(drop=True)
    log(f"\n[2] Unique promoter sequences: {len(uniq)} / {len(full)} rows", fh)
    log(f"    -> Each unique sequence appears on avg "
        f"{len(full)/len(uniq):.2f} times across (tissue, stage, variety) "
        f"combinations.", fh)

    # Exact-match cross-split check
    seq_to_splits = defaultdict(set)
    for s, sp in zip(full["seq_upper"], full["split"]):
        seq_to_splits[s].add(sp)
    exact_cross = {s: sp for s, sp in seq_to_splits.items() if len(sp) > 1}
    log(f"\n[3] Exact-sequence cross-split matches: {len(exact_cross)} unique "
        f"sequences (of {len(seq_to_splits)})", fh)
    ex_rows = []
    for s, sp in exact_cross.items():
        rows = full[full["seq_upper"] == s]
        ex_rows.append({
            "seq_hash": hashlib.md5(s.encode()).hexdigest()[:10],
            "n_rows_total": len(rows),
            "splits": ",".join(sorted(sp)),
            "n_train": int((rows["split"] == "TRAIN").sum()),
            "n_dev": int((rows["split"] == "DEV").sum()),
            "n_test": int((rows["split"] == "TEST").sum()),
            "label_values": ",".join(sorted(map(str, rows["label"].unique()))),
        })
    ex_df = pd.DataFrame(ex_rows)
    ex_df.to_csv(os.path.join(OUT_DIR, "exact_duplicates_by_split.csv"), index=False)
    if len(ex_df) > 0:
        log(f"    Of these, {(ex_df['n_train']>0).sum()} are in train, "
            f"{(ex_df['n_test']>0).sum()} in test.", fh)

    # CD-HIT-EST
    uniq["fa_id"] = uniq["split"] + "_" + uniq.index.map(lambda i: f"{i:06d}")
    uniq_to_rows = full.groupby("seq_upper")
    write_fasta(uniq, FASTA_UNIQ, id_col="fa_id")
    log(f"\n[4] Wrote {len(uniq)} unique sequences to {FASTA_UNIQ}", fh)

    log(f"\n[5] Running cd-hit-est at identity={IDENTITY}", fh)
    run_cdhit(FASTA_UNIQ, CLSTR_PREFIX)
    clusters = parse_clstr(CLSTR_PREFIX + ".clstr")
    log(f"    -> {len(clusters)} clusters formed from {len(uniq)} unique seqs", fh)

    id_to_origin_split = dict(zip(uniq["fa_id"], uniq["split"]))
    id_to_seq = dict(zip(uniq["fa_id"], uniq["seq_upper"]))

    mixed_rows = []
    mixed_count = 0
    for cid, members in enumerate(clusters):
        if len(members) < 2:
            continue
        splits = set()
        per_split = {"TRAIN": 0, "DEV": 0, "TEST": 0}
        for m in members:
            seq = id_to_seq[m]
            rows = uniq_to_rows.get_group(seq)
            for sp in rows["split"].unique():
                splits.add(sp)
            for sp in rows["split"]:
                per_split[sp] += 1
        if len(splits) > 1:
            mixed_count += 1
            mixed_rows.append({
                "cluster_id": cid,
                "n_members_unique_seqs": len(members),
                "n_rows_train": per_split["TRAIN"],
                "n_rows_dev": per_split["DEV"],
                "n_rows_test": per_split["TEST"],
                "splits": ",".join(sorted(splits)),
                "example_members": ";".join(members[:5]),
            })
    mixed_df = pd.DataFrame(mixed_rows)
    mixed_df.to_csv(os.path.join(OUT_DIR, "leakage_clusters.csv"), index=False)

    multi_member = sum(1 for c in clusters if len(c) > 1)
    log(f"\n[6] Near-duplicate cluster analysis", fh)
    log(f"    Multi-member clusters: {multi_member}", fh)
    log(f"    Clusters spanning >1 split (LEAKAGE):  {mixed_count}", fh)

    test_rows_in_mixed = int(mixed_df["n_rows_test"].sum()) if len(mixed_df) else 0
    train_rows_in_mixed = int(mixed_df["n_rows_train"].sum()) if len(mixed_df) else 0
    N_TEST = split_counts.get("TEST", 1)
    log(f"\n[7] Impact on evaluation integrity", fh)
    log(f"    Test rows in clusters also containing TRAIN: {test_rows_in_mixed}"
        f"  ({test_rows_in_mixed/N_TEST*100:.2f}% of test)", fh)
    log(f"    Train rows in such clusters:                {train_rows_in_mixed}", fh)

    summary = pd.DataFrame([{
        "total_rows": len(full),
        "train_rows": split_counts.get("TRAIN", 0),
        "dev_rows": split_counts.get("DEV", 0),
        "test_rows": N_TEST,
        "unique_sequences": len(uniq),
        "exact_match_cross_split_seqs": len(ex_df),
        "cdhit_identity_threshold": IDENTITY,
        "cdhit_total_clusters": len(clusters),
        "cdhit_multi_member_clusters": multi_member,
        "cdhit_mixed_split_clusters": mixed_count,
        "test_rows_in_mixed_clusters": test_rows_in_mixed,
        "train_rows_in_mixed_clusters": train_rows_in_mixed,
        "pct_test_at_risk": round(test_rows_in_mixed / max(1, N_TEST) * 100, 3),
    }])
    summary.to_csv(os.path.join(OUT_DIR, "leakage_summary.csv"), index=False)

    log("\n[8] CONCLUSION", fh)
    log("-" * 72, fh)
    pct = test_rows_in_mixed / max(1, N_TEST) * 100
    if pct == 0:
        verdict = "CLEAN"
        log("  ZERO leakage detected at 80% sequence identity.", fh)
    elif pct < 5:
        verdict = "MINOR"
        log(f"  {pct:.2f}% of test rows share a >=80%-identical sequence with training.", fh)
    else:
        verdict = "LEAKAGE"
        log(f"  {pct:.2f}% of test rows share a >=80%-identical sequence with training.", fh)
    log(f"\n  Verdict: {verdict}", fh)
    log(f"\nAll outputs in {OUT_DIR}", fh)
    fh.close()


if __name__ == "__main__":
    main()
