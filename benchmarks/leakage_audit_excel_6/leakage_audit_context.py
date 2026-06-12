#!/usr/bin/env python3
"""Context-aware leakage audit for excel_6 (conflicting labels removed).
Observation unit = (promoter_sequence, Variety, Tissue, Stage).
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

INPUT = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
OUT_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/benchmarks/leakage_audit_excel_6"
CLSTR = os.path.join(OUT_DIR, "cdhit80.clstr")
FA = os.path.join(OUT_DIR, "unique_sequences.fasta")
LOG = os.path.join(OUT_DIR, "leakage_audit_context.log")

CONTEXT = ["Variety", "Tissue", "Stage"]


def log(msg, fh): print(msg); fh.write(msg + "\n")


def reproduce_split():
    df = pd.read_csv(INPUT)
    df["label"] = (df["Gene Regulation"] == "Up Regulated").astype(int)
    df = df.dropna(subset=["promoter_sequence"]).reset_index(drop=True)
    df["seq"] = df["promoter_sequence"].str.upper()
    df["tuple"] = (
        df["seq"] + "|" + df["Variety"].astype(str)
        + "|" + df["Tissue"].astype(str) + "|" + df["Stage"].astype(str)
    )
    tr, tmp = train_test_split(df, test_size=0.2, random_state=42,
                               stratify=df["label"])
    dv, te = train_test_split(tmp, test_size=0.5, random_state=42,
                              stratify=tmp["label"])
    tr = tr.assign(split="TRAIN"); dv = dv.assign(split="DEV"); te = te.assign(split="TEST")
    return pd.concat([tr, dv, te], ignore_index=True), tr, dv, te


def parse_cdhit():
    fa_to_seq = {}
    cur = None
    with open(FA) as f:
        for ln in f:
            if ln.startswith(">"):
                cur = ln[1:].strip()
            else:
                fa_to_seq[cur] = ln.strip()
    seq_to_cid = {}
    with open(CLSTR) as f:
        cid = -1
        for ln in f:
            if ln.startswith(">Cluster"):
                cid += 1
            else:
                parts = ln.strip().split(">")
                if len(parts) > 1:
                    m = parts[1].split("...")[0]
                    if m in fa_to_seq:
                        seq_to_cid[fa_to_seq[m]] = cid
    return seq_to_cid


def main():
    fh = open(LOG, "w")
    log("=" * 72, fh)
    log("CONTEXT-AWARE LEAKAGE AUDIT", fh)
    log("INPUT: RiceMetaSys_Drought_merged_all_excels_6.csv", fh)
    log("(conflicting-label rows removed)", fh)
    log("Observation unit = (sequence, Variety, Tissue, Stage)", fh)
    log("=" * 72, fh)

    full, tr, dv, te = reproduce_split()
    N_TR, N_DV, N_TE = len(tr), len(dv), len(te)
    log(f"\nSplit sizes: train={N_TR}  dev={N_DV}  test={N_TE}", fh)
    log(f"Total rows: {len(full)}", fh)

    # Within-corpus duplicates
    log("\n[0] Within-corpus duplication (all rows)", fh)
    dup_tuple = full["tuple"].duplicated(keep=False).sum()
    dup_seq_only = full["seq"].duplicated(keep=False).sum()
    log(f"    Rows sharing exact (seq, Variety, Tissue, Stage) tuple: "
        f"{dup_tuple}  ({dup_tuple/len(full)*100:.2f}%)", fh)
    log(f"    Rows sharing only sequence (different context): "
        f"{dup_seq_only}  ({dup_seq_only/len(full)*100:.2f}%)", fh)
    log(f"    Unique tuples:     {full['tuple'].nunique()}", fh)
    log(f"    Unique sequences:  {full['seq'].nunique()}", fh)

    tup_lab = full.groupby("tuple")["label"].nunique()
    noisy_tuples = (tup_lab > 1).sum()
    log(f"    Tuples with conflicting labels across duplicates: "
        f"{noisy_tuples}  (label noise)", fh)

    train_tuples = set(tr["tuple"]); dev_tuples = set(dv["tuple"])
    train_seqs = set(tr["seq"]); dev_seqs = set(dv["seq"])
    pool_tuples = train_tuples | dev_tuples
    pool_seqs = train_seqs | dev_seqs

    # L1: tuple-level leakage
    log("\n[L1] TUPLE-level leakage  (real duplicates across splits)", fh)
    te_tuple_tr = te["tuple"].isin(train_tuples).values
    te_tuple_pool = te["tuple"].isin(pool_tuples).values
    log(f"    Test rows with exact (seq, V, T, S) tuple in TRAIN:        "
        f"{te_tuple_tr.sum()}/{N_TE} ({te_tuple_tr.mean()*100:.2f}%)", fh)
    log(f"    Test rows with exact (seq, V, T, S) tuple in TRAIN or DEV: "
        f"{te_tuple_pool.sum()}/{N_TE} ({te_tuple_pool.mean()*100:.2f}%)", fh)

    # L3: sequence-only overlap
    log("\n[L3] SEQUENCE-ONLY overlap  (seq shared, context differs)", fh)
    te_seq_pool = te["seq"].isin(pool_seqs).values
    te_seq_pool_but_new_tuple = te_seq_pool & ~te_tuple_pool
    log(f"    Test rows with sequence seen in pool but new (V, T, S):   "
        f"{te_seq_pool_but_new_tuple.sum()}/{N_TE} "
        f"({te_seq_pool_but_new_tuple.mean()*100:.2f}%)", fh)
    log("    -> NOT inherently leakage for a CONTEXT-AWARE classifier.", fh)

    seq_lab_counts = full.groupby("seq")["label"].nunique()
    single_label_seqs = set(seq_lab_counts[seq_lab_counts == 1].index)
    te_seq_single_in_pool = te["seq"].isin(pool_seqs & single_label_seqs).values
    te_memorizable_no_context = te_seq_single_in_pool & ~te_tuple_pool
    log(f"    Of those, seq always has the SAME label in pool "
        f"(memorization possible):", fh)
    log(f"       {te_memorizable_no_context.sum()}/{N_TE} "
        f"({te_memorizable_no_context.mean()*100:.2f}%) of test rows", fh)

    flip_seqs = set(seq_lab_counts[seq_lab_counts > 1].index)
    te_flip_in_pool = te["seq"].isin(pool_seqs & flip_seqs).values & ~te_tuple_pool
    log(f"    Of those, seq has FLIPPING labels in pool (context truly needed):", fh)
    log(f"       {te_flip_in_pool.sum()}/{N_TE} "
        f"({te_flip_in_pool.mean()*100:.2f}%) of test rows", fh)

    # L4: near-duplicate
    log("\n[L4] NEAR-DUPLICATE sequence (>=80% identity)", fh)
    seq_to_cid = parse_cdhit()
    cluster_has = defaultdict(lambda: {"TRAIN": set(), "DEV": set(), "TEST": set()})
    for s, sp, t in zip(full["seq"], full["split"], full["tuple"]):
        c = seq_to_cid.get(s)
        if c is not None:
            cluster_has[c][sp].add(t)

    te_near = np.zeros(N_TE, dtype=bool)
    for i, s in enumerate(te["seq"].values):
        c = seq_to_cid.get(s)
        if c is None: continue
        if cluster_has[c]["TRAIN"] or cluster_has[c]["DEV"]:
            te_near[i] = True
    log(f"    Test rows whose CD-HIT cluster also contains a pool row: "
        f"{te_near.sum()}/{N_TE} ({te_near.mean()*100:.2f}%)", fh)
    add_near = te_near & ~te_seq_pool
    log(f"    Additional to exact-sequence overlap: "
        f"{add_near.sum()}/{N_TE} ({add_near.mean()*100:.2f}%)", fh)

    # Clean subsets
    clean_tuple = ~te_tuple_pool
    clean_seq = ~te_seq_pool
    clean_near = ~te_near
    log("\n[5] CLEAN TEST SUBSETS (for sensitivity analysis)", fh)
    log(f"    No tuple-level leakage (drop L1): "
        f"{clean_tuple.sum()}/{N_TE} ({clean_tuple.mean()*100:.2f}%)", fh)
    log(f"    No seq overlap at all (drop L1+L3): "
        f"{clean_seq.sum()}/{N_TE} ({clean_seq.mean()*100:.2f}%)", fh)
    log(f"    No seq cluster overlap (drop L1+L3+L4): "
        f"{clean_near.sum()}/{N_TE} ({clean_near.mean()*100:.2f}%)", fh)

    # Save per-row flags
    te = te.copy()
    te["tuple_in_pool"] = te_tuple_pool
    te["seq_in_pool_new_context"] = te_seq_pool_but_new_tuple
    te["seq_in_pool_single_label"] = te_memorizable_no_context
    te["seq_in_pool_flip_label"] = te_flip_in_pool
    te["near80_in_pool"] = te_near
    te["clean_tuple"] = clean_tuple
    te["clean_seq"] = clean_seq
    te["clean_near"] = clean_near
    te[["seq", "Variety", "Tissue", "Stage", "label",
        "tuple_in_pool", "seq_in_pool_new_context",
        "seq_in_pool_single_label", "seq_in_pool_flip_label",
        "near80_in_pool", "clean_tuple", "clean_seq", "clean_near"]].to_csv(
        os.path.join(OUT_DIR, "test_row_leakage_flags_context.csv"), index=False)

    pd.DataFrame([{
        "total_rows": len(full),
        "unique_tuples": int(full["tuple"].nunique()),
        "unique_sequences": int(full["seq"].nunique()),
        "noisy_tuples_conflicting_labels": int(noisy_tuples),
        "test_L1_tuple_in_pool": int(te_tuple_pool.sum()),
        "pct_test_L1_tuple_in_pool": round(te_tuple_pool.mean()*100, 3),
        "test_L3_seq_in_pool_new_context": int(te_seq_pool_but_new_tuple.sum()),
        "pct_test_L3": round(te_seq_pool_but_new_tuple.mean()*100, 3),
        "test_L3_single_label_seq": int(te_memorizable_no_context.sum()),
        "test_L3_flip_label_seq": int(te_flip_in_pool.sum()),
        "test_L4_near80_in_pool": int(te_near.sum()),
        "pct_test_L4": round(te_near.mean()*100, 3),
        "test_clean_tuple": int(clean_tuple.sum()),
        "test_clean_seq": int(clean_seq.sum()),
        "test_clean_near": int(clean_near.sum()),
    }]).to_csv(os.path.join(OUT_DIR, "leakage_summary_context.csv"), index=False)

    # Conclusion
    pct_L1 = te_tuple_pool.mean() * 100
    pct_mem = te_memorizable_no_context.mean() * 100
    pct_ctx = te_flip_in_pool.mean() * 100

    log("\n" + "=" * 72, fh)
    log("CORRECTED CONCLUSION  (context-aware)", fh)
    log("=" * 72, fh)

    if pct_L1 == 0:
        v1 = "CLEAN — no tuple-level leakage"
    elif pct_L1 < 1:
        v1 = "NEGLIGIBLE tuple leakage"
    elif pct_L1 < 5:
        v1 = "MINOR tuple leakage — disclose"
    else:
        v1 = "MATERIAL tuple leakage — fix split"

    log(f"\n  L1 verdict: {v1}  —  {pct_L1:.2f}% of test", fh)
    log(f"\n  L3 verdict (sequence overlap with different context):", fh)
    log(f"      {te_seq_pool_but_new_tuple.sum()} rows "
        f"({te_seq_pool_but_new_tuple.mean()*100:.2f}% of test)", fh)
    log(f"      Breakdown:", fh)
    log(f"        - single-label sequence (memorizable w/o context): {pct_mem:.2f}%", fh)
    log(f"        - flipping-label sequence (context truly needed):  {pct_ctx:.2f}%", fh)

    log(f"\n  All outputs in {OUT_DIR}", fh)
    fh.close()


if __name__ == "__main__":
    main()
