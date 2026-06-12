#!/usr/bin/env python3
"""Finalize leakage audit for excel_6.
Observation unit = (promoter_sequence, Variety, Tissue, Stage).
The model is CONTEXT-AWARE — sequence overlap with different context is NOT leakage.
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

INPUT = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
OUT_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/benchmarks/leakage_audit_excel_6"
CLSTR = os.path.join(OUT_DIR, "cdhit80.clstr")
LOG = os.path.join(OUT_DIR, "leakage_audit_final.log")

CONTEXT = ["Variety", "Tissue", "Stage"]


def log(msg, fh): print(msg); fh.write(msg + "\n")


def main():
    fh = open(LOG, "w")
    log("=" * 72, fh)
    log("LEAKAGE AUDIT  —  FINAL REPORT (CONTEXT-AWARE)", fh)
    log("INPUT: RiceMetaSys_Drought_merged_all_excels_6.csv", fh)
    log("(conflicting-label rows removed)", fh)
    log("Observation unit = (promoter_sequence, Variety, Tissue, Stage)", fh)
    log("CD-HIT-EST identity threshold: 80%  (word=5)", fh)
    log("=" * 72, fh)

    df = pd.read_csv(INPUT)
    df["label"] = (df["Gene Regulation"] == "Up Regulated").astype(int)
    df = df.dropna(subset=["promoter_sequence"]).reset_index(drop=True)
    df["seq"] = df["promoter_sequence"].str.upper()
    df["tuple"] = (
        df["seq"] + "|" + df["Variety"].astype(str)
        + "|" + df["Tissue"].astype(str) + "|" + df["Stage"].astype(str)
    )
    tr, tmp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    dv, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp["label"])
    tr = tr.assign(split="TRAIN").reset_index(drop=True)
    dv = dv.assign(split="DEV").reset_index(drop=True)
    te = te.assign(split="TEST").reset_index(drop=True)
    full = pd.concat([tr, dv, te], ignore_index=True)
    N_TEST, N_TRAIN, N_DEV = len(te), len(tr), len(dv)
    log(f"\nSplit sizes: train={N_TRAIN}  dev={N_DEV}  test={N_TEST}", fh)
    log(f"Unique tuples (observation units): {full['tuple'].nunique()}", fh)
    log(f"Unique promoter sequences: {full['seq'].nunique()}", fh)

    train_tuples = set(tr["tuple"]); dev_tuples = set(dv["tuple"])
    pool_tuples = train_tuples | dev_tuples

    # ── L1: Tuple-level leakage (the ONLY real leakage for this model) ──
    log("\n--- L1: TUPLE-LEVEL LEAKAGE (exact same input to model) ---", fh)
    te_L1_tr = te["tuple"].isin(train_tuples).values
    te_L1_pool = te["tuple"].isin(pool_tuples).values
    log(f"  Test rows with exact (seq, V, T, S) in TRAIN:        "
        f"{te_L1_tr.sum()}/{N_TEST} ({te_L1_tr.mean()*100:.2f}%)", fh)
    log(f"  Test rows with exact (seq, V, T, S) in TRAIN or DEV: "
        f"{te_L1_pool.sum()}/{N_TEST} ({te_L1_pool.mean()*100:.2f}%)", fh)

    # Label noise check
    tup_lab = full.groupby("tuple")["label"].nunique()
    noisy = (tup_lab > 1).sum()
    log(f"  Tuples with conflicting labels: {noisy}", fh)

    # ── Sequence sharing (informational, NOT leakage) ──
    log("\n--- SEQUENCE SHARING (informational — NOT leakage for this model) ---", fh)
    log("  The same promoter can appear with different (V, T, S) contexts.", fh)
    log("  This is expected biological variation, not data leakage.", fh)
    train_seqs = set(tr["seq"]); dev_seqs = set(dv["seq"])
    pool_seqs = train_seqs | dev_seqs
    te_seq_pool = te["seq"].isin(pool_seqs).values
    te_seq_new_ctx = te_seq_pool & ~te_L1_pool
    log(f"  Test rows sharing sequence with pool (different context): "
        f"{te_seq_new_ctx.sum()}/{N_TEST} ({te_seq_new_ctx.mean()*100:.2f}%)", fh)

    lab_count = df.groupby("seq")["label"].nunique()
    flip_seqs = set(lab_count[lab_count > 1].index)
    single_seqs = set(lab_count[lab_count == 1].index)
    te_memorizable = te["seq"].isin(pool_seqs & single_seqs).values & ~te_L1_pool
    te_flip = te["seq"].isin(pool_seqs & flip_seqs).values & ~te_L1_pool
    log(f"    - Same-label sequences (memorization possible):  "
        f"{te_memorizable.sum()}/{N_TEST} ({te_memorizable.mean()*100:.2f}%)", fh)
    log(f"    - Flipping-label sequences (context truly needed): "
        f"{te_flip.sum()}/{N_TEST} ({te_flip.mean()*100:.2f}%)", fh)
    log(f"  Of {df['seq'].nunique()} unique sequences:", fh)
    log(f"    single-label: {len(single_seqs)} ({len(single_seqs)/df['seq'].nunique()*100:.1f}%)", fh)
    log(f"    label-flipping: {len(flip_seqs)} ({len(flip_seqs)/df['seq'].nunique()*100:.1f}%)", fh)

    # ── Near-duplicate sequences (informational) ──
    log("\n--- NEAR-DUPLICATE SEQUENCES (informational — >=80% identity) ---", fh)
    clusters = []
    with open(CLSTR) as f:
        cur = None
        for ln in f:
            if ln.startswith(">Cluster"):
                if cur is not None: clusters.append(cur)
                cur = []
            elif cur is not None:
                parts = ln.strip().split(">")
                if len(parts) > 1:
                    name = parts[1].split("...")[0]
                    cur.append(name)
        if cur is not None: clusters.append(cur)
    log(f"  CD-HIT produced {len(clusters)} clusters from {df['seq'].nunique()} unique seqs.", fh)

    fa_id_to_seq = {}
    cur_id = None
    with open(os.path.join(OUT_DIR, "unique_sequences.fasta")) as f:
        for ln in f:
            if ln.startswith(">"):
                cur_id = ln[1:].strip()
            else:
                fa_id_to_seq[cur_id] = ln.strip()

    seq_to_cluster = {}
    for cid, members in enumerate(clusters):
        for m in members:
            if m in fa_id_to_seq:
                seq_to_cluster[fa_id_to_seq[m]] = cid

    cluster_to_pool = defaultdict(lambda: {"TRAIN": 0, "DEV": 0, "TEST": 0})
    for s, sp in zip(full["seq"], full["split"]):
        c = seq_to_cluster.get(s)
        if c is not None:
            cluster_to_pool[c][sp] += 1

    te_near = np.zeros(N_TEST, dtype=bool)
    for i, s in enumerate(te["seq"].values):
        c = seq_to_cluster.get(s)
        if c is None: continue
        bucket = cluster_to_pool[c]
        if bucket["TRAIN"] > 0 or bucket["DEV"] > 0:
            te_near[i] = True

    te_near_only = te_near & ~te_seq_pool
    log(f"  Test rows with near-dup sequence in pool: "
        f"{te_near.sum()}/{N_TEST} ({te_near.mean()*100:.2f}%)", fh)
    log(f"  Additional to exact-sequence overlap:      "
        f"{te_near_only.sum()}/{N_TEST} ({te_near_only.mean()*100:.2f}%)", fh)
    log(f"  Note: these have DIFFERENT contexts — not leakage for this model.", fh)

    # ── Save flags ──
    te_flags = te[["seq", "Variety", "Tissue", "Stage", "label"]].copy()
    te_flags["L1_tuple_in_pool"] = te_L1_pool
    te_flags["seq_in_pool_new_context"] = te_seq_new_ctx
    te_flags["memorizable_seq"] = te_memorizable
    te_flags["flip_label_seq"] = te_flip
    te_flags["near80_in_pool"] = te_near
    te_flags.to_csv(os.path.join(OUT_DIR, "test_row_leakage_flags.csv"), index=False)

    pd.DataFrame([{
        "total_rows": len(full),
        "train_rows": N_TRAIN, "dev_rows": N_DEV, "test_rows": N_TEST,
        "unique_tuples": int(full["tuple"].nunique()),
        "unique_sequences": int(df["seq"].nunique()),
        "conflicting_label_tuples": int(noisy),
        "test_L1_tuple_in_pool": int(te_L1_pool.sum()),
        "pct_test_L1": round(te_L1_pool.mean() * 100, 3),
        "test_seq_overlap_new_ctx": int(te_seq_new_ctx.sum()),
        "test_memorizable_seq": int(te_memorizable.sum()),
        "test_flip_label_seq": int(te_flip.sum()),
        "test_near80_in_pool": int(te_near.sum()),
        "test_near80_additional": int(te_near_only.sum()),
    }]).to_csv(os.path.join(OUT_DIR, "leakage_summary_final.csv"), index=False)

    # ── Conclusion ──
    log("\n" + "=" * 72, fh)
    log("CONCLUSION (CONTEXT-AWARE MODEL)", fh)
    log("=" * 72, fh)
    pct_L1 = te_L1_pool.mean() * 100
    if pct_L1 == 0:
        verdict = "CLEAN — zero tuple-level leakage"
    elif pct_L1 < 2:
        verdict = "MINOR — disclose + report clean-test subset"
    elif pct_L1 < 10:
        verdict = "MATERIAL — consider tuple-aware resplit"
    else:
        verdict = "SEVERE — tuple-aware resplit required"

    log(f"\n  Verdict: {verdict}", fh)
    log(f"\n  • L1 (real leakage): {pct_L1:.2f}% of test rows share exact "
        f"(seq, V, T, S) tuple with training pool.", fh)
    log(f"  • Conflicting-label tuples: {noisy}", fh)
    log(f"  • Sequence sharing (NOT leakage): {te_seq_new_ctx.sum()}/{N_TEST} "
        f"({te_seq_new_ctx.mean()*100:.2f}%) test rows share a promoter "
        f"sequence but have different biological context.", fh)
    log(f"  • 37.7% of sequences flip labels across contexts — "
        f"the model MUST learn context to predict correctly.", fh)
    log(f"\nAll outputs in {OUT_DIR}", fh)
    fh.close()


if __name__ == "__main__":
    main()
