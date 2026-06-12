#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positional Motif Analysis Pipeline for Rice Drought Response
=============================================================
This pipeline extracts POSITIONAL motif features from promoter sequences.

Features Extracted:
1. Binned counts - Split promoter into 5 bins of 200bp each
2. Relative position - Average position of each motif (0=distal, 1=TSS)
3. First/Last occurrence - Position of first and last motif occurrence
4. Density gradient - Motif density in proximal vs distal regions
5. Motif co-occurrence patterns - Pairs of motifs that appear together
6. Global counts (from V3) - Total count of each motif

Output Features (~400+ features):
- 54 motifs × 5 bins = 270 binned features
- 54 motifs × 1 = 54 relative position features
- 54 motifs × 2 = 108 first/last position features
- 54 motifs × 1 = 54 density gradient features
- Top 50 co-occurrence pairs = 50 features
- 54 global counts + 4 seq stats = 58 features
- Context features (Variety, Tissue, Stage one-hot)

@author: saiful
"""

import os
import re
import warnings
from typing import Dict, Any, Tuple, List, Set
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
# OUTPUT_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/output/"
OUTPUT_DIR = "/home/saiful/CARCM/output/"

RANDOM_SEED = 42

# Positional feature settings
NUM_BINS = 5  # Number of positional bins
BIN_SIZE = 200  # Expected bin size in bp (promoters ~1000bp)
PROMOTER_LENGTH = 1000  # Standard promoter length for normalization

# Co-occurrence settings
TOP_COOCCURRENCE_PAIRS = 50  # Number of top co-occurring motif pairs to include


# =============================================================================
# MOTIF DEFINITIONS (regex patterns)
# =============================================================================

MOTIFS = {
    # ABA-responsive elements
    'ABRE': 'ACGTG[GT][CT]',
    'ABRE_core': 'ACGTG',
    'CE1': 'TGCCACCGG',
    'CE3': 'ACGCGTGTC',
    
    # Dehydration-responsive elements
    'DRE_CRT': 'CCGAC',
    'DRE_full': '[AG]CCGAC',
    'LTRE': 'CCGAC',
    
    # MYB binding sites
    'MYB_core': 'C[ACGT]GTT[AG]',
    'MYB1AT': '[AT]AACCA',
    'MYB2': 'TAACTG',
    'MYBCORE': 'C[ACGT]GTT[AG]',
    'MYC': 'CA[ACGT][ACGT]TG',
    
    # W-box (WRKY binding)
    'WBOX': 'TTGAC[CT]',
    'WBOX_core': 'TGAC',
    
    # GT elements
    'GT1': 'G[AG][AT]AA[AT]',
    'GT1_core': 'GAAAAA',
    
    # Heat shock elements
    'HSE': 'GAA[ACGT][ACGT]TTC',
    'HSE_core': '[ACGT]GAA[ACGT]',
    
    # Ethylene responsive
    'GCC_box': 'GCCGCC',
    'ERE': 'A[AT]TTCAAA',
    
    # Light/circadian elements
    'GBOX': 'CACGTG',
    'IBOX': 'GATAAG',
    'CAAT_box': 'CCAAT',
    
    # Stress responsive
    'STRE': 'AGGGG',
    'LTRE_like': 'ACCGACA',
    
    # Wound/pathogen response
    'JASE': 'CGTCA',
    'TBOX': 'ACTTTG',
    
    # NAC binding sites
    'NAC_core': 'CACG',
    'NACRS': 'CATGTG',
    
    # bZIP binding sites
    'ACGT_core': 'ACGT',
    'ABRE_like': '[AC]ACGT[GC][CGT]',
    
    # Auxin responsive
    'AuxRE': 'TGTCTC',
    'AuxRE_rev': 'GAGACA',
    
    # Root specific
    'ROOTMOTIF': 'ATATT',
    
    # Additional elements
    'AS1': 'TGACG',
    'TATA_box': 'TATAAA',
    'CACTFTPPCA1': '[CT]ACT',
    'ICE_box': 'CA[ACGT][ACGT]TG',
    'CBF_DRE': 'GTCGAC',
    
    # Simple 4-mers (common regulatory elements)
    'ACGT': 'ACGT',
    'CACG': 'CACG',
    'CATG': 'CATG',
    'CGTG': 'CGTG',
    'GCGT': 'GCGT',
    'AACA': 'AACA',
    'CAAC': 'CAAC',
    'AAAG': 'AAAG',
    'GATA': 'GATA',
    'TATA': 'TATA',
    'CAAT': 'CAAT',
    'GCCG': 'GCCG',
    'CGCG': 'CGCG',
    'TGAC': 'TGAC',
    'GTAC': 'GTAC',
}

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
        # Convert position to forward strand coordinates
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
    gc_skew = (g - c) / (g + c + 1.0)  # +1 for stability

    return length, round(gc, 4), round(at, 4), round(gc_skew, 6)


# =============================================================================
# POSITIONAL FEATURE EXTRACTION
# =============================================================================

def extract_positional_features(sequence: str, num_bins: int = 5) -> Dict[str, Any]:
    """
    Extract all positional motif features from a promoter sequence.
    
    Features:
    1. Binned counts - motif counts in each bin
    2. Relative position - normalized average position (0=distal, 1=TSS)
    3. First/Last occurrence - position of first and last motif
    4. Density gradient - proximal density / distal density
    5. Global count - total count
    
    Returns dict with all features.
    """
    if pd.isna(sequence) or not isinstance(sequence, str) or len(sequence) < 100:
        # Return zeros for empty/invalid sequences
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
    all_positions = {}  # Store positions for co-occurrence analysis
    
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
        
        # 2. Relative position (normalized to 0-1, where 1 = near TSS/3' end)
        if total_count > 0:
            avg_pos = np.mean(positions) / seq_len
            features[f"{motif_name}_rel_pos"] = round(avg_pos, 4)
        else:
            features[f"{motif_name}_rel_pos"] = 0.5  # Default to middle
        
        # 3. First and Last occurrence positions
        if total_count > 0:
            features[f"{motif_name}_first_pos"] = round(positions[0] / seq_len, 4)
            features[f"{motif_name}_last_pos"] = round(positions[-1] / seq_len, 4)
        else:
            features[f"{motif_name}_first_pos"] = 0.5
            features[f"{motif_name}_last_pos"] = 0.5
        
        # 4. Density gradient (proximal vs distal)
        # Proximal = last 2 bins (near TSS), Distal = first 2 bins
        proximal_count = sum(bin_counts[-2:]) if num_bins >= 2 else bin_counts[-1]
        distal_count = sum(bin_counts[:2]) if num_bins >= 2 else bin_counts[0]
        
        # Gradient: positive = more proximal, negative = more distal
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
    """
    Calculate motif co-occurrence features.
    Two motifs co-occur if they appear within 'window_size' bp of each other.
    
    Returns counts of co-occurring motif pairs.
    """
    cooccurrence = defaultdict(int)
    
    motif_names = list(all_positions.keys())
    
    for i, motif1 in enumerate(motif_names):
        for motif2 in motif_names[i:]:  # Include self-pairs
            if motif1 == motif2:
                # Self-pair: count pairs of same motif within window
                positions = all_positions[motif1]
                for p1_idx, p1 in enumerate(positions):
                    for p2 in positions[p1_idx+1:]:
                        if abs(p2 - p1) <= window_size:
                            cooccurrence[f"cooc_{motif1}_{motif1}"] += 1
            else:
                # Different motifs
                pos1 = all_positions[motif1]
                pos2 = all_positions[motif2]
                for p1 in pos1:
                    for p2 in pos2:
                        if abs(p1 - p2) <= window_size:
                            # Use sorted names for consistent key
                            key = f"cooc_{'_'.join(sorted([motif1, motif2]))}"
                            cooccurrence[key] += 1
    
    return dict(cooccurrence)


def find_top_cooccurrence_pairs(df: pd.DataFrame, n_pairs: int = 50) -> List[str]:
    """
    Analyze dataset to find the most common co-occurring motif pairs.
    Returns list of top n_pairs feature names.
    """
    print(f"\nFinding top {n_pairs} co-occurring motif pairs...")
    
    cooccurrence_counts = defaultdict(int)
    
    for idx, row in df.iterrows():
        seq = row['promoter_sequence']
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) < 100:
            continue
        
        # Get positions for all motifs
        all_positions = {}
        for motif_name, pattern in MOTIFS.items():
            all_positions[motif_name] = find_motif_positions(seq, pattern)
        
        # Count co-occurrences
        cooc = calculate_cooccurrence_features(all_positions, len(seq))
        for key, count in cooc.items():
            if count > 0:
                cooccurrence_counts[key] += 1
    
    # Sort by frequency and get top pairs
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
    """Load and clean the rice drought dataset."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Convert Gene Regulation -> binary label
    df["label"] = (df["Gene Regulation"] == "Up Regulated").astype(int)

    # Drop missing promoter sequences
    df = df.dropna(subset=["promoter_sequence"])

    # Remove very short sequences (< 100 bp)
    df = df[df["promoter_sequence"].astype(str).str.len() >= 100]

    print(f"  Cleaned shape: {df.shape}")

    print("\n  Variety distribution:")
    print(df["Variety"].value_counts())

    print("\n  Tissue distribution:")
    print(df["Tissue"].value_counts())

    print("\n  Stage distribution:")
    print(df["Stage"].value_counts())

    print("\n  Label distribution (0=Down, 1=Up):")
    print(df["label"].value_counts())
    
    # Promoter length distribution
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
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{total} sequences ({(idx+1)/total*100:.1f}%)...")
        
        seq = row['promoter_sequence']
        
        # Extract positional features
        pos_features, all_positions = extract_positional_features(seq, NUM_BINS)
        
        # Calculate co-occurrence features
        if pd.notna(seq) and isinstance(seq, str) and len(seq) >= 100:
            cooc_features = calculate_cooccurrence_features(all_positions, len(seq))
        else:
            cooc_features = {}
        
        # Add only top co-occurrence pairs
        for pair_name in top_cooc_pairs:
            pos_features[pair_name] = cooc_features.get(pair_name, 0)
        
        # Add sequence statistics
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
    """Run the positional motif feature extraction pipeline."""
    print("="*70)
    print("POSITIONAL MOTIF ANALYSIS PIPELINE")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_and_clean_data(INPUT_FILE)
    
    # Find top co-occurrence pairs (sample for speed)
    sample_df = df.sample(min(5000, len(df)), random_state=RANDOM_SEED)
    top_cooc_pairs = find_top_cooccurrence_pairs(sample_df, TOP_COOCCURRENCE_PAIRS)
    
    # Save co-occurrence pairs for later use
    with open(os.path.join(OUTPUT_DIR, 'top_cooccurrence_pairs.txt'), 'w') as f:
        for pair in top_cooc_pairs:
            f.write(pair + '\n')
    
    # Create positional features
    positional_df = create_positional_features(df, top_cooc_pairs)
    
    # Create context features
    context_df, variety_cols, tissue_cols, stage_cols = create_context_features(df)
    
    # Combine all features
    final_df = pd.concat([positional_df, context_df], axis=1)
    final_df['label'] = df['label'].values
    
    print(f"\nFinal feature matrix shape: {final_df.shape}")
    
    # Feature summary
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    
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
    
    # Split data (stratified by label)
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    
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
    
    # Save datasets
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(OUTPUT_DIR, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    # Save feature names
    feature_names = [c for c in final_df.columns if c != 'label']
    with open(os.path.join(OUTPUT_DIR, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(name + '\n')
    
    # Save feature group info
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
    
    print(f"  Saved train.csv, dev.csv, test.csv to {OUTPUT_DIR}")
    print(f"  Saved feature_names.txt and feature_groups.json")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
