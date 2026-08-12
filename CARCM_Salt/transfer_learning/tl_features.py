#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transfer Learning — Step 1: build a SHARED feature space for drought + salt.
============================================================================
Drought and salt use different motif sets and different Variety/Tissue/Stage
vocabularies, so the encoder cannot be transferred unless both datasets are
represented in an IDENTICAL feature space. This script:

  1. Uses a SHARED motif dictionary = union(drought motifs, salinity motifs)
     (60 motifs; drought definition wins on the single key collision, MYB2).
  2. Computes top-50 co-occurrence pairs ONCE on the (larger) drought data and
     applies the same pair columns to both datasets.
  3. One-hot encodes context over the UNION vocabulary (Variety=17, Tissue=8,
     Stage=6) so both datasets share identical context columns. NOTE: drought
     and salt share NO varieties, but salt's tissues/stages are subsets of
     drought's — so tissue/stage embeddings + the motif encoder are what
     transfer; salt-variety embeddings are learned fresh at fine-tune time.

Outputs (transfer_learning/data/):
    drought_features.parquet, salt_features.parquet   (identical columns)
    shared_feature_groups.json, context_categories.json, shared_motifs.json

Reuses the tested feature-extraction functions from rice_positional_pipeline_salt
by monkey-patching its MOTIFS / MOTIF_NAMES / NUM_BINS globals (read-only import;
the original salt pipeline file is NOT modified).
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/saiful/DeepRice_Salt/CARCM_Salt")
import rice_positional_pipeline_salt as pl  # noqa: E402

DROUGHT_6 = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
SALT_6 = "/home/saiful/DeepRice_Salt/data_preprocess/RiceMetaSys_Salt_merged_all_excels_6.csv"
OUT_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/transfer_learning/data"
NUM_BINS = 5
TOP_COOC = 50
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# --- original drought motif set (from CARCM rice_positional_pipeline.py) ---
DROUGHT_MOTIFS = {
    'ABRE': 'ACGTG[GT][CT]', 'ABRE_core': 'ACGTG', 'CE1': 'TGCCACCGG', 'CE3': 'ACGCGTGTC',
    'DRE_CRT': 'CCGAC', 'DRE_full': '[AG]CCGAC', 'LTRE': 'CCGAC',
    'MYB_core': 'C[ACGT]GTT[AG]', 'MYB1AT': '[AT]AACCA', 'MYB2': 'TAACTG',
    'MYBCORE': 'C[ACGT]GTT[AG]', 'MYC': 'CA[ACGT][ACGT]TG', 'WBOX': 'TTGAC[CT]',
    'WBOX_core': 'TGAC', 'GT1': 'G[AG][AT]AA[AT]', 'GT1_core': 'GAAAAA',
    'HSE': 'GAA[ACGT][ACGT]TTC', 'HSE_core': '[ACGT]GAA[ACGT]', 'GCC_box': 'GCCGCC',
    'ERE': 'A[AT]TTCAAA', 'GBOX': 'CACGTG', 'IBOX': 'GATAAG', 'CAAT_box': 'CCAAT',
    'STRE': 'AGGGG', 'LTRE_like': 'ACCGACA', 'JASE': 'CGTCA', 'TBOX': 'ACTTTG',
    'NAC_core': 'CACG', 'NACRS': 'CATGTG', 'ACGT_core': 'ACGT', 'ABRE_like': '[AC]ACGT[GC][CGT]',
    'AuxRE': 'TGTCTC', 'AuxRE_rev': 'GAGACA', 'ROOTMOTIF': 'ATATT', 'AS1': 'TGACG',
    'TATA_box': 'TATAAA', 'CACTFTPPCA1': '[CT]ACT', 'ICE_box': 'CA[ACGT][ACGT]TG',
    'CBF_DRE': 'GTCGAC', 'ACGT': 'ACGT', 'CACG': 'CACG', 'CATG': 'CATG', 'CGTG': 'CGTG',
    'GCGT': 'GCGT', 'AACA': 'AACA', 'CAAC': 'CAAC', 'AAAG': 'AAAG', 'GATA': 'GATA',
    'TATA': 'TATA', 'CAAT': 'CAAT', 'GCCG': 'GCCG', 'CGCG': 'CGCG', 'TGAC': 'TGAC', 'GTAC': 'GTAC',
}


def build_shared_motifs():
    shared = dict(DROUGHT_MOTIFS)               # drought definitions win on overlap
    for k, v in pl.SALINITY_MOTIFS.items():     # add salt-only motifs
        shared.setdefault(k, v)
    return shared


def encode_context(df, varieties, tissues, stages):
    cols = {}
    for v in varieties:
        cols[f"Variety_{v}"] = (df["Variety"].values == v).astype(np.int8)
    for t in tissues:
        cols[f"Tissue_{t}"] = (df["Tissue"].values == t).astype(np.int8)
    for s in stages:
        cols[f"Stage_{s}"] = (df["Stage"].values == s).astype(np.int8)
    return pd.DataFrame(cols)


def feature_groups_from_columns(df):
    binned = [c for c in df.columns if "_bin" in c]
    relpos = [c for c in df.columns if "_rel_pos" in c]
    firstpos = [c for c in df.columns if "_first_pos" in c]
    lastpos = [c for c in df.columns if "_last_pos" in c]
    grad = [c for c in df.columns if "_density_grad" in c]
    count = [c for c in df.columns if c.endswith("_count")]
    cooc = [c for c in df.columns if c.startswith("cooc_")]
    seq_stats = ["length", "gc_content", "at_content", "gc_skew"]
    variety = [c for c in df.columns if c.startswith("Variety_")]
    tissue = [c for c in df.columns if c.startswith("Tissue_")]
    stage = [c for c in df.columns if c.startswith("Stage_")]
    return {
        "binned": binned, "relative_position": relpos, "first_position": firstpos,
        "last_position": lastpos, "density_gradient": grad, "global_count": count,
        "cooccurrence": cooc, "seq_stats": seq_stats,
        "variety": variety, "tissue": tissue, "stage": stage,
    }


def save_table(df, path_base):
    try:
        df.to_parquet(path_base + ".parquet", index=False)
        return path_base + ".parquet"
    except Exception as e:
        print(f"  parquet failed ({e}); writing csv")
        df.to_csv(path_base + ".csv", index=False)
        return path_base + ".csv"


def main():
    print("=" * 70)
    print("TRANSFER LEARNING — SHARED FEATURE SPACE (drought + salt)")
    print("=" * 70)

    shared = build_shared_motifs()
    pl.MOTIFS = shared
    pl.MOTIF_NAMES = list(shared.keys())
    pl.NUM_BINS = NUM_BINS
    print(f"Shared motif set: {len(shared)} motifs "
          f"(drought {len(DROUGHT_MOTIFS)} ∪ salinity {len(pl.SALINITY_MOTIFS)})")

    print("\n--- Loading drought ---")
    dro = pl.load_and_clean_data(DROUGHT_6).reset_index(drop=True)
    print("\n--- Loading salt ---")
    sal = pl.load_and_clean_data(SALT_6).reset_index(drop=True)

    # top co-occurrence pairs from drought (source domain), applied to both
    print("\n--- Top co-occurrence pairs (from drought) ---")
    sample = dro.sample(min(5000, len(dro)), random_state=RANDOM_SEED)
    top_cooc = pl.find_top_cooccurrence_pairs(sample, TOP_COOC)

    print("\n--- Extracting drought positional features (this is the slow step) ---")
    dro_pos = pl.create_positional_features(dro, top_cooc)
    print("\n--- Extracting salt positional features ---")
    sal_pos = pl.create_positional_features(sal, top_cooc)

    # Normalise drought's data-provenance suffixes before the union, so a
    # cultivar present in both modules maps to ONE level. Without this,
    # "IR64_Cleaned" (drought) and "IR64" (salt) are distinct strings and the
    # shared IR64 cultivar is split across two embeddings, so nothing learned
    # about IR64 during drought pretraining reaches salt's IR64 rows.
    VARIETY_RENAME = {"IR64_Cleaned": "IR64", "N22_Cleaned3": "N22"}
    dro["Variety"] = dro["Variety"].replace(VARIETY_RENAME)
    sal["Variety"] = sal["Variety"].replace(VARIETY_RENAME)
    shared_vars = sorted(set(dro["Variety"]) & set(sal["Variety"]))
    print(f"\nCultivars shared by drought and salt after normalisation: {shared_vars}")

    # shared context vocabulary (union)
    varieties = sorted(set(dro["Variety"]) | set(sal["Variety"]))
    tissues = sorted(set(dro["Tissue"]) | set(sal["Tissue"]))
    stages = sorted(set(dro["Stage"]) | set(sal["Stage"]))
    print(f"\nShared context vocab: Variety={len(varieties)} "
          f"Tissue={len(tissues)} Stage={len(stages)}")

    dro_ctx = encode_context(dro, varieties, tissues, stages)
    sal_ctx = encode_context(sal, varieties, tissues, stages)

    dro_final = pd.concat([dro_pos.reset_index(drop=True), dro_ctx], axis=1)
    dro_final["label"] = dro["label"].values
    sal_final = pd.concat([sal_pos.reset_index(drop=True), sal_ctx], axis=1)
    sal_final["label"] = sal["label"].values

    # ensure identical column order
    assert list(dro_final.columns) == list(sal_final.columns), "column mismatch!"
    print(f"\nDrought feature matrix: {dro_final.shape}")
    print(f"Salt    feature matrix: {sal_final.shape}")

    groups = feature_groups_from_columns(dro_final)

    pd_dro = save_table(dro_final, os.path.join(OUT_DIR, "drought_features"))
    pd_sal = save_table(sal_final, os.path.join(OUT_DIR, "salt_features"))
    with open(os.path.join(OUT_DIR, "shared_feature_groups.json"), "w") as f:
        json.dump(groups, f, indent=2)
    with open(os.path.join(OUT_DIR, "context_categories.json"), "w") as f:
        json.dump({"varieties": varieties, "tissues": tissues, "stages": stages}, f, indent=2)
    with open(os.path.join(OUT_DIR, "shared_motifs.json"), "w") as f:
        json.dump(shared, f, indent=2)
    with open(os.path.join(OUT_DIR, "top_cooccurrence_pairs.txt"), "w") as f:
        f.write("\n".join(top_cooc))

    print(f"\nSaved:\n  {pd_dro}\n  {pd_sal}\n  shared_feature_groups.json, "
          f"context_categories.json, shared_motifs.json")
    print(f"  n_motifs (shared) = {len(groups['global_count'])}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
