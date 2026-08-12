#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess RiceMetaSys_Salt_merged_all_excels.csv into
RiceMetaSys_Salt_merged_all_excels_6.csv, replaying the EXACT same steps that
produced /home/saiful/DeepRice/.../RiceMetaSys_Drought_merged_all_excels_6.csv.

Pipeline (each stage writes its own CSV, mirroring the Drought pipeline):

  raw  -> _2 : select [RAP ID, Tissue, Stage, Variety, Gene Regulation],
              dropna(any), fix Tissue/Stage spelling variants
  _2   -> _3 : map RAP ID -> promoter_sequence via 1kb-upstream FASTA
              (RAP 'g'->'t' first occurrence; prefer '-01' transcript variant)
  _3   -> _4 : dropna(any), keep [promoter_sequence, Variety, Tissue, Stage,
              Gene Regulation]
  _4   -> _5 : drop exact duplicate rows
  _5   -> _6 : drop ambiguous rows -- any (promoter_sequence, Variety, Tissue,
              Stage) combo mapping to >1 Gene Regulation label is removed entirely

Final _6 has the same column layout as the Drought _6:
    promoter_sequence, Variety, Tissue, Stage, Gene Regulation

Run inside the `dnabert` conda environment:
    conda activate dnabert
    python preprocess_salt_pipeline.py
"""
import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths
RAW   = "/home/saiful/DeepRice_Salt/RiceMetaSys_Salt/RiceMetaSys_Salt_merged_all_excels.csv"
FASTA = "/home/saiful/DeepRice/IRGSP-1.0_1kb-upstream_2025-03-19.fasta"
OUTDIR = "/home/saiful/DeepRice_Salt/data_preprocess"

P2 = os.path.join(OUTDIR, "RiceMetaSys_Salt_merged_all_excels_2.csv")
P3 = os.path.join(OUTDIR, "RiceMetaSys_Salt_merged_all_excels_3.csv")
P4 = os.path.join(OUTDIR, "RiceMetaSys_Salt_merged_all_excels_4.csv")
P5 = os.path.join(OUTDIR, "RiceMetaSys_Salt_merged_all_excels_5.csv")
P6 = os.path.join(OUTDIR, "RiceMetaSys_Salt_merged_all_excels_6.csv")

os.makedirs(OUTDIR, exist_ok=True)


# =========================================================================== Step 1: raw -> _2
def step1_select_and_clean():
    df = pd.read_csv(RAW)
    print(f"[1] raw loaded: {df.shape}")

    new_df = df[["RAP ID", "Tissue", "Stage", "Variety", "Gene Regulation"]].copy()

    before = len(new_df)
    clean_df = new_df.dropna().copy()
    print(f"[1] dropna: {before} -> {len(clean_df)} (dropped {before - len(clean_df)})")

    # Tissue spelling fixes (no-ops if a variant isn't present)
    tissue_fix = {"Anth1er": "Anther", "FlagLeaf": "Flag Leaf", "Leaf1": "Leaf", "Root1": "Root"}
    clean_df["Tissue"] = clean_df["Tissue"].replace(tissue_fix)

    # Stage spelling fixes
    clean_df["Stage"] = clean_df["Stage"].replace({"VegetativeStage": "Vegetative Stage"})

    clean_df.to_csv(P2, index=False)
    print(f"[1] saved {P2}  shape={clean_df.shape}")
    return clean_df


# =========================================================================== Step 2: _2 -> _3 (FASTA join)
def read_fasta_simple(path):
    seqs, cid, cur = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cid is not None:
                    seqs[cid] = "".join(cur)
                cid = line[1:].split()[0]
                cur = []
            elif line:
                cur.append(line)
        if cid is not None:
            seqs[cid] = "".join(cur)
    return seqs


def pick_best_variant(entries):
    """entries: list of (full_id, seq) for the same base transcript id."""
    for fid, seq in entries:
        if fid.endswith("-01"):
            return fid, seq

    def suffix_num(fid):
        m = re.search(r"-(\d+)$", fid)
        return int(m.group(1)) if m else 10**9

    return sorted(entries, key=lambda x: suffix_num(x[0]))[0]


def step2_add_promoter(df):
    seq_dict = read_fasta_simple(FASTA)
    print(f"[2] FASTA loaded: {len(seq_dict):,} sequences")

    base_to_entries = {}
    for fid, seq in seq_dict.items():
        base_to_entries.setdefault(fid.split("-")[0], []).append((fid, seq))

    df_new = df.copy()
    # RAP ID  Os01g0123400 -> transcript Os01t0123400 (first 'g' -> 't')
    df_new["transcript_id"] = (
        df_new["RAP ID"].astype(str).str.strip().str.replace(r"g", "t", n=1, regex=True)
    )

    def get_promoter(tid):
        if tid in base_to_entries:
            return pick_best_variant(base_to_entries[tid])[1]
        return np.nan

    df_new["promoter_sequence"] = df_new["transcript_id"].apply(get_promoter)

    matched, total = df_new["promoter_sequence"].notna().sum(), len(df_new)
    print(f"[2] promoter matched: {matched:,}/{total:,} ({matched/total:.2%})")

    df_new.to_csv(P3, index=False)
    print(f"[2] saved {P3}  shape={df_new.shape}")
    return df_new


# =========================================================================== Step 3: _3 -> _4
def step3_select_model_cols(df):
    before = df.shape
    df_clean = df.dropna(how="any").reset_index(drop=True)
    print(f"[3] dropna: {before} -> {df_clean.shape}")

    selected = ["promoter_sequence", "Variety", "Tissue", "Stage", "Gene Regulation"]
    df_model = df_clean[selected].copy()
    df_model.to_csv(P4, index=False)
    print(f"[3] saved {P4}  shape={df_model.shape}")
    return df_model


# =========================================================================== Step 4: _4 -> _5
def step4_drop_duplicates(df):
    df_unique = df.drop_duplicates()
    print(f"[4] drop_duplicates: {df.shape} -> {df_unique.shape}")
    df_unique.to_csv(P5, index=False)
    print(f"[4] saved {P5}  shape={df_unique.shape}")
    return df_unique


# =========================================================================== Step 5: _5 -> _6
def step5_drop_ambiguous(df):
    keys4 = ["promoter_sequence", "Variety", "Tissue", "Stage"]
    n_labels = df.groupby(keys4)["Gene Regulation"].transform("nunique")
    df_unamb = df[n_labels == 1].reset_index(drop=True)
    removed = len(df) - len(df_unamb)
    print(f"[5] drop ambiguous label groups: removed {removed} rows "
          f"({df.shape} -> {df_unamb.shape})")
    df_unamb.to_csv(P6, index=False)
    print(f"[5] saved {P6}  shape={df_unamb.shape}")
    return df_unamb


# =========================================================================== main
if __name__ == "__main__":
    d2 = step1_select_and_clean()
    d3 = step2_add_promoter(d2)
    d4 = step3_select_model_cols(d3)
    d5 = step4_drop_duplicates(d4)
    d6 = step5_drop_ambiguous(d5)

    print("\n=== FINAL (_6) ===")
    print("columns:", d6.columns.tolist())
    print("shape  :", d6.shape)
    print("Gene Regulation distribution:")
    print(d6["Gene Regulation"].value_counts())
    print("\nFirst 3 rows:")
    print(d6.assign(promoter_sequence=d6["promoter_sequence"].str[:40] + "...").head(3).to_string(index=False))
