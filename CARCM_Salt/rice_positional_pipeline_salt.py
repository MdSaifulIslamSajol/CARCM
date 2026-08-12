#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positional Motif Analysis Pipeline for Rice SALINITY/SALT Response
==================================================================
Salt-adapted copy of CARCM's rice_positional_pipeline.py.

ONLY two things differ from the drought pipeline:
  1. MOTIFS -> replaced with a curated set of SALINITY/SALT-stress
     cis-regulatory elements (see SALINITY_MOTIFS below).
  2. INPUT_FILE / OUTPUT_DIR -> point to the Salt dataset and CARCM_Salt.

Everything else (binned counts, relative/first/last position, density
gradient, co-occurrence, sequence stats, context one-hot, stratified
train/dev/test split) is byte-for-byte identical to the drought pipeline,
so the resulting feature matrix is directly comparable.

@author: saiful  (salt adaptation)
"""

import os
import re
import sys
import warnings
from typing import Dict, Any, Tuple, List
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

warnings.filterwarnings("ignore")


# =============================================================================
# CONSOLE LOGGING  —  mirror all stdout/stderr to a plain-text log file
# =============================================================================
CONSOLE_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "console_output_rice_positional_pipeline_salt.txt",
)


class _Tee:
    """Duplicate a stream to the original console and a log file."""

    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, data):
        self.console.write(data)
        self.logfile.write(data)

    def flush(self):
        self.console.flush()
        self.logfile.flush()


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "/home/saiful/DeepRice_Salt/data_preprocess/RiceMetaSys_Salt_merged_all_excels_6.csv"
OUTPUT_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/output/"

RANDOM_SEED = 42

# Positional feature settings
NUM_BINS = 5            # Number of positional bins
BIN_SIZE = 200          # Expected bin size in bp (promoters ~1000bp)
PROMOTER_LENGTH = 1000  # Standard promoter length for normalization

# Co-occurrence settings
TOP_COOCCURRENCE_PAIRS = 50  # Number of top co-occurring motif pairs to include


# =============================================================================
# MOTIF DEFINITIONS  —  SALINITY / SALT-STRESS cis-regulatory elements
# =============================================================================
# Curated from the salt/osmotic-stress promoter literature. Salt stress is
# largely ABA- and osmotic-driven, so the ABA/DRE/MYB/NAC core is retained
# (these are the canonical salinity-responsive elements), and salt-specific
# regulators are added: CAMTA/CG-box (Ca2+ signalling), Dof, GT-1 (salt-
# inducible), as-1/ocs (salt+SA crosstalk), STRE (general ionic stress).
# Purely drought/heat-only elements (e.g. ERE/GCC ethylene, CBF_DRE,
# ICE_box, AuxRE, TBOX, JASE) were dropped in favour of salt-relevant ones.

SALINITY_MOTIFS = {
    # --- ABA-responsive elements (salt response is largely ABA-mediated) ---
    'ABRE':       'ACGTG[GT][CT]',     # ABA-responsive element
    'ABRE_core':  'ACGTG',
    'ABRE_like':  '[AC]ACGT[GC][CGT]',
    'CE1':        'TGCCACCGG',         # coupling element 1 (ABRE partner)
    'CE3':        'ACGCGTGTC',         # coupling element 3
    'GBOX':       'CACGTG',            # G-box (bZIP/ABA)
    'ACGT_core':  'ACGT',

    # --- Dehydration / osmotic (DRE/CRT) — core salt + drought element ---
    'DRE_CRT':    'CCGAC',
    'DRE_full':   '[AG]CCGAC',
    'CRT_core':   'CCGAC',
    'LTRE':       'CCGAC',             # low-temp/osmotic, overlaps salt

    # --- MYB / MYC (ABA-dependent salt signalling) ---
    'MYB_core':   'C[ACGT]GTT[AG]',
    'MYB1AT':     '[AT]AACCA',
    'MYB2':       '[CT]AAC[GT]G',      # MYB2 (dehydration/salt)
    'MYBCORE':    'C[ACGT]GTT[AG]',
    'MBS':        'CAACTG',            # MYB binding site (salt/drought)
    'MYC':        'CA[ACGT][ACGT]TG',  # MYCATERD1 (CANNTG)
    'MYCATRD22':  'CACATG',            # MYC (RD22, ABA/salt)

    # --- NAC (SNAC salt-stress master regulators) ---
    'NAC_core':   'CACG',
    'NACRS':      'CATGTG',            # NAC recognition sequence

    # --- WRKY / W-box (salt-responsive WRKY TFs) ---
    'WBOX':       'TTGAC[CT]',
    'WBOX_core':  'TGAC',

    # --- bZIP / as-1 / ocs (salt & salicylate crosstalk) ---
    'AS1':        'TGACG',
    'OCS':        'TGACGT[CA]A',       # ocs-element

    # --- GT-1 element (salt-inducible, e.g. SCaM-4 / CaMV) ---
    'GT1':        'G[AG][AT]AA[AT]',
    'GT1_core':   'GAAAAA',

    # --- Ca2+ / CAMTA signalling (salt stress) ---
    'CGCG_box':   '[AC]CGCG[CGT]',     # CAMTA / CG-1 box
    'CGCG':       'CGCG',

    # --- General stress-response element (ionic/osmotic) ---
    'STRE':       'AGGGG',

    # --- Dof zinc-finger (salt-responsive Dof) ---
    'DOF_core':   'AAAG',

    # --- Heat-shock (cross-stress under combined salt stress) ---
    'HSE':        'GAA[ACGT][ACGT]TTC',
    'HSE_core':   '[ACGT]GAA[ACGT]',

    # --- GATA / light context ---
    'GATA':       'GATA',
    'IBOX':       'GATAAG',
    'CAAT_box':   'CCAAT',

    # --- Root / tissue context (salt first sensed in root) ---
    'ROOTMOTIF':  'ATATT',
    'CACTFTPPCA1': '[CT]ACT',

    # --- Core promoter ---
    'TATA_box':   'TATAAA',

    # --- Simple regulatory k-mers (generic signal, mirrors CARCM drought) ---
    'CATG':       'CATG',
    'CGTG':       'CGTG',
    'GCGT':       'GCGT',
    'AACA':       'AACA',
    'CAAC':       'CAAC',
    'TATA':       'TATA',
    'CAAT':       'CAAT',
    'GCCG':       'GCCG',
    'GTAC':       'GTAC',
}

# Keep the public name `MOTIFS` so the rest of the pipeline is unchanged.
MOTIFS = SALINITY_MOTIFS
MOTIF_NAMES = list(MOTIFS.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def find_motif_positions(sequence: str, pattern: str) -> List[int]:
    """
    Find all positions where a motif occurs (both strands).
    Returns list of start positions (0-indexed from 5' end).
    """
    if pd.isna(sequence) or not isinstance(sequence, str) or len(sequence) == 0:
        return []

    seq = sequence.upper()
    positions = []

    # Forward strand
    for match in re.finditer(pattern, seq):
        positions.append(match.start())

    # Reverse complement strand (position is still relative to forward)
    rev_seq = reverse_complement(seq)
    for match in re.finditer(pattern, rev_seq):
        pos = len(seq) - match.end()
        positions.append(pos)

    return sorted(set(positions))


def get_seq_stats(sequence: str) -> Tuple[int, float, float, float]:
    """Calculate sequence statistics: length, gc%, at%, gc_skew."""
    if pd.isna(sequence) or not isinstance(sequence, str) or len(sequence) == 0:
        return 0, 0.0, 0.0, 0.0
    seq = sequence.upper()
    length = len(seq)
    g = seq.count("G")
    c = seq.count("C")
    a = seq.count("A")
    t = seq.count("T")

    gc = (g + c) / length * 100.0 if length else 0.0
    at = (a + t) / length * 100.0 if length else 0.0
    gc_skew = (g - c) / (g + c + 1.0)

    return length, round(gc, 4), round(at, 4), round(gc_skew, 6)


# =============================================================================
# POSITIONAL FEATURE EXTRACTION
# =============================================================================

def extract_positional_features(sequence: str, num_bins: int = 5) -> Dict[str, Any]:
    """Extract all positional motif features from a promoter sequence."""
    if pd.isna(sequence) or not isinstance(sequence, str) or len(sequence) < 100:
        features = {}
        for motif in MOTIF_NAMES:
            for b in range(num_bins):
                features[f"{motif}_bin{b}"] = 0
            features[f"{motif}_rel_pos"] = 0.5
            features[f"{motif}_first_pos"] = 0.5
            features[f"{motif}_last_pos"] = 0.5
            features[f"{motif}_density_grad"] = 0.0
            features[f"{motif}_count"] = 0
        return features

    seq = sequence.upper()
    seq_len = len(seq)
    bin_size = seq_len / num_bins

    features = {}
    all_positions = {}

    for motif_name, pattern in MOTIFS.items():
        positions = find_motif_positions(seq, pattern)
        all_positions[motif_name] = positions
        total_count = len(positions)

        # 1. Binned counts
        bin_counts = [0] * num_bins
        for pos in positions:
            bin_idx = min(int(pos / bin_size), num_bins - 1)
            bin_counts[bin_idx] += 1
        for b in range(num_bins):
            features[f"{motif_name}_bin{b}"] = bin_counts[b]

        # 2. Relative position (0=distal, 1=TSS)
        if total_count > 0:
            features[f"{motif_name}_rel_pos"] = round(np.mean(positions) / seq_len, 4)
        else:
            features[f"{motif_name}_rel_pos"] = 0.5

        # 3. First and last occurrence
        if total_count > 0:
            features[f"{motif_name}_first_pos"] = round(positions[0] / seq_len, 4)
            features[f"{motif_name}_last_pos"] = round(positions[-1] / seq_len, 4)
        else:
            features[f"{motif_name}_first_pos"] = 0.5
            features[f"{motif_name}_last_pos"] = 0.5

        # 4. Density gradient (proximal vs distal)
        proximal_count = sum(bin_counts[-2:]) if num_bins >= 2 else bin_counts[-1]
        distal_count = sum(bin_counts[:2]) if num_bins >= 2 else bin_counts[0]
        if proximal_count + distal_count > 0:
            gradient = (proximal_count - distal_count) / (proximal_count + distal_count)
        else:
            gradient = 0.0
        features[f"{motif_name}_density_grad"] = round(gradient, 4)

        # 5. Global count
        features[f"{motif_name}_count"] = total_count

    return features, all_positions


def calculate_cooccurrence_features(all_positions: Dict[str, List[int]],
                                    seq_len: int,
                                    window_size: int = 100) -> Dict[str, int]:
    """Calculate motif co-occurrence features within a bp window."""
    cooccurrence = defaultdict(int)
    motif_names = list(all_positions.keys())

    for i, motif1 in enumerate(motif_names):
        for motif2 in motif_names[i:]:
            if motif1 == motif2:
                positions = all_positions[motif1]
                for p1_idx, p1 in enumerate(positions):
                    for p2 in positions[p1_idx + 1:]:
                        if abs(p2 - p1) <= window_size:
                            cooccurrence[f"cooc_{motif1}_{motif1}"] += 1
            else:
                pos1 = all_positions[motif1]
                pos2 = all_positions[motif2]
                for p1 in pos1:
                    for p2 in pos2:
                        if abs(p1 - p2) <= window_size:
                            key = f"cooc_{'_'.join(sorted([motif1, motif2]))}"
                            cooccurrence[key] += 1

    return dict(cooccurrence)


def find_top_cooccurrence_pairs(df: pd.DataFrame, n_pairs: int = 50) -> List[str]:
    """Find the most common co-occurring motif pairs across the dataset."""
    print(f"\nFinding top {n_pairs} co-occurring motif pairs...")
    cooccurrence_counts = defaultdict(int)

    for idx, row in df.iterrows():
        seq = row['promoter_sequence']
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) < 100:
            continue
        all_positions = {}
        for motif_name, pattern in MOTIFS.items():
            all_positions[motif_name] = find_motif_positions(seq, pattern)
        cooc = calculate_cooccurrence_features(all_positions, len(seq))
        for key, count in cooc.items():
            if count > 0:
                cooccurrence_counts[key] += 1

    sorted_pairs = sorted(cooccurrence_counts.items(), key=lambda x: x[1], reverse=True)
    top_pairs = [pair[0] for pair in sorted_pairs[:n_pairs]]

    print(f"  Top 10 co-occurring pairs:")
    for pair, count in sorted_pairs[:10]:
        print(f"    {pair}: {count} sequences")

    return top_pairs


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and clean the rice SALT dataset."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")

    df.columns = df.columns.str.strip()
    df["label"] = (df["Gene Regulation"] == "Up Regulated").astype(int)
    df = df.dropna(subset=["promoter_sequence"])
    df = df[df["promoter_sequence"].astype(str).str.len() >= 100]
    print(f"  Cleaned shape: {df.shape}")

    print("\n  Variety distribution:");  print(df["Variety"].value_counts())
    print("\n  Tissue distribution:");   print(df["Tissue"].value_counts())
    print("\n  Stage distribution:");    print(df["Stage"].value_counts())
    print("\n  Label distribution (0=Down, 1=Up):"); print(df["label"].value_counts())

    df['_seq_len'] = df['promoter_sequence'].str.len()
    print(f"\n  Promoter length: min={df['_seq_len'].min()}, max={df['_seq_len'].max()}, "
          f"mean={df['_seq_len'].mean():.0f}, median={df['_seq_len'].median():.0f}")
    return df


def create_positional_features(df: pd.DataFrame, top_cooc_pairs: List[str]) -> pd.DataFrame:
    """Create all positional motif features for the dataset."""
    print(f"\nExtracting positional features for {len(df)} sequences...")
    print(f"  Features: {NUM_BINS} bins × {len(MOTIFS)} motifs + positions + gradients + co-occurrence")

    all_features = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{total} sequences ({(idx+1)/total*100:.1f}%)...")
        seq = row['promoter_sequence']
        pos_features, all_positions = extract_positional_features(seq, NUM_BINS)

        if pd.notna(seq) and isinstance(seq, str) and len(seq) >= 100:
            cooc_features = calculate_cooccurrence_features(all_positions, len(seq))
        else:
            cooc_features = {}
        for pair_name in top_cooc_pairs:
            pos_features[pair_name] = cooc_features.get(pair_name, 0)

        length, gc, at, skew = get_seq_stats(seq)
        pos_features['length'] = length
        pos_features['gc_content'] = gc
        pos_features['at_content'] = at
        pos_features['gc_skew'] = skew
        all_features.append(pos_features)

    features_df = pd.DataFrame(all_features)
    print(f"  Created {features_df.shape[1]} features")
    return features_df


def create_context_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """One-hot encode context variables (Variety, Tissue, Stage)."""
    print("\nCreating context features...")
    variety_dummies = pd.get_dummies(df['Variety'], prefix='Variety')
    tissue_dummies = pd.get_dummies(df['Tissue'], prefix='Tissue')
    stage_dummies = pd.get_dummies(df['Stage'], prefix='Stage')

    variety_cols = list(variety_dummies.columns)
    tissue_cols = list(tissue_dummies.columns)
    stage_cols = list(stage_dummies.columns)

    print(f"  Varieties: {len(variety_cols)}")
    print(f"  Tissues: {len(tissue_cols)}")
    print(f"  Stages: {len(stage_cols)}")

    context_df = pd.concat([variety_dummies, tissue_dummies, stage_dummies], axis=1)
    return context_df, variety_cols, tissue_cols, stage_cols


def main():
    print("=" * 70)
    print("POSITIONAL MOTIF ANALYSIS PIPELINE  —  SALT / SALINITY")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_clean_data(INPUT_FILE)

    sample_df = df.sample(min(5000, len(df)), random_state=RANDOM_SEED)
    top_cooc_pairs = find_top_cooccurrence_pairs(sample_df, TOP_COOCCURRENCE_PAIRS)

    with open(os.path.join(OUTPUT_DIR, 'top_cooccurrence_pairs.txt'), 'w') as f:
        for pair in top_cooc_pairs:
            f.write(pair + '\n')

    positional_df = create_positional_features(df, top_cooc_pairs)
    context_df, variety_cols, tissue_cols, stage_cols = create_context_features(df)

    # align indices before concat (positional_df is RangeIndex; reset context)
    context_df = context_df.reset_index(drop=True)
    final_df = pd.concat([positional_df, context_df], axis=1)
    final_df['label'] = df['label'].values

    print(f"\nFinal feature matrix shape: {final_df.shape}")

    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    binned_cols = [c for c in final_df.columns if '_bin' in c]
    relpos_cols = [c for c in final_df.columns if '_rel_pos' in c]
    firstpos_cols = [c for c in final_df.columns if '_first_pos' in c]
    lastpos_cols = [c for c in final_df.columns if '_last_pos' in c]
    gradient_cols = [c for c in final_df.columns if '_density_grad' in c]
    count_cols = [c for c in final_df.columns if c.endswith('_count')]
    cooc_cols = [c for c in final_df.columns if c.startswith('cooc_')]
    seq_stat_cols = ['length', 'gc_content', 'at_content', 'gc_skew']

    print(f"  Binned counts:      {len(binned_cols)} features")
    print(f"  Relative positions: {len(relpos_cols)} features")
    print(f"  First positions:    {len(firstpos_cols)} features")
    print(f"  Last positions:     {len(lastpos_cols)} features")
    print(f"  Density gradients:  {len(gradient_cols)} features")
    print(f"  Global counts:      {len(count_cols)} features")
    print(f"  Co-occurrence:      {len(cooc_cols)} features")
    print(f"  Sequence stats:     {len(seq_stat_cols)} features")
    print(f"  Context (Variety):  {len(variety_cols)} features")
    print(f"  Context (Tissue):   {len(tissue_cols)} features")
    print(f"  Context (Stage):    {len(stage_cols)} features")
    print(f"  TOTAL:              {final_df.shape[1] - 1} features (excluding label)")

    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    train_df, temp_df = train_test_split(
        final_df, test_size=0.2, random_state=RANDOM_SEED, stratify=final_df['label']
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['label']
    )
    print(f"  Train: {len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%)")
    print(f"  Dev:   {len(dev_df)} ({len(dev_df)/len(final_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)")
    print(f"\n  Label distribution:")
    print(f"    Train: Up={train_df['label'].sum()} Down={len(train_df) - train_df['label'].sum()}")
    print(f"    Dev:   Up={dev_df['label'].sum()} Down={len(dev_df) - dev_df['label'].sum()}")
    print(f"    Test:  Up={test_df['label'].sum()} Down={len(test_df) - test_df['label'].sum()}")

    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(OUTPUT_DIR, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

    feature_names = [c for c in final_df.columns if c != 'label']
    with open(os.path.join(OUTPUT_DIR, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(name + '\n')

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
        'stage': stage_cols
    }
    import json
    with open(os.path.join(OUTPUT_DIR, 'feature_groups.json'), 'w') as f:
        json.dump(feature_groups, f, indent=2)

    # record the motif set used (for provenance / classifier n_motifs)
    with open(os.path.join(OUTPUT_DIR, 'motif_definitions.json'), 'w') as f:
        json.dump(MOTIFS, f, indent=2)

    print(f"  Saved train.csv, dev.csv, test.csv to {OUTPUT_DIR}")
    print(f"  Saved feature_names.txt, feature_groups.json, motif_definitions.json")
    print(f"  N motifs (salinity): {len(MOTIFS)}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    with open(CONSOLE_LOG_FILE, "w") as _logf:
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(_orig_stdout, _logf)
        sys.stderr = _Tee(_orig_stderr, _logf)
        try:
            main()
        finally:
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
    print(f"\nConsole output saved to: {CONSOLE_LOG_FILE}")
