import pandas as pd

# Option A: Use the full absolute path (if you're on your own Linux/Mac machine)
file_path = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_2.csv"

# Option B: If you're already in the project folder → relative path is usually cleaner
# file_path = "data_preprocess/RiceMetaSys_Drought_merged_all_excels_2.csv"

df = pd.read_csv(file_path)
print("File loaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())
#%%
def read_fasta_simple(fasta_path):
    """
    Reads a FASTA file and returns a dictionary of {header: sequence}.
    - header = everything after '>' until first space (usually the ID)
    - sequence = concatenated string of all following lines until next header
    Handles multiline sequences automatically.
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.rstrip()  # remove trailing \n
                
                if line.startswith('>'):
                    # Save previous sequence if any
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    
                    # New header → take ID (first word after >)
                    header_parts = line[1:].split()
                    current_id = header_parts[0]  # e.g. Os01g0100100 or locus name
                    current_seq = []
                else:
                    # Sequence line → add (ignore empty lines)
                    if line:
                        current_seq.append(line)
            
            # Don't forget the last sequence
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    except FileNotFoundError:
        print(f"Error: File not found → {fasta_path}")
        print("Current working directory:", os.getcwd())
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}

# ────────────────────────────────────────────────
#                  Usage example
# ────────────────────────────────────────────────

import os

fasta_path = "/home/saiful/DeepRice/IRGSP-1.0_1kb-upstream_2025-03-19.fasta"

# Quick path check
print("File exists?", os.path.exists(fasta_path))
print("Is file?", os.path.isfile(fasta_path) if os.path.exists(fasta_path) else "—")

# Load
seq_dict = read_fasta_simple(fasta_path)

if seq_dict:
    print(f"\nSuccessfully loaded {len(seq_dict):,} promoter sequences")
    
    # Show summary
    print("First few entries:")
    for i, (seq_id, seq) in enumerate(seq_dict.items(), 1):
        if i > 5:
            break
        print(f"  {i}. {seq_id:20}   len = {len(seq):,} bp")
        print(f"     First 60 bp: {seq[:60]}...")
    
    # Optional: total bases
    total_bases = sum(len(s) for s in seq_dict.values())
    print(f"\nTotal bases across all sequences: {total_bases:,}")
else:
    print("No sequences loaded — check path or file permissions.")
    
# %%
import re
import numpy as np

# -----------------------------
# 1) Build a fast lookup from FASTA headers
#    Example FASTA key: Os02t0216300-01
#    We map base transcript: Os02t0216300 -> list of (full_id, seq)
# -----------------------------
base_to_entries = {}

for full_id, seq in seq_dict.items():
    # base_id = part before first '-'  (Os02t0216300 from Os02t0216300-01)
    base_id = full_id.split("-")[0]
    base_to_entries.setdefault(base_id, []).append((full_id, seq))

# -----------------------------
# 2) A helper to choose which transcript variant to use
#    Priority:
#      (a) if "-01" exists, use it
#      (b) else use the smallest numeric suffix
#      (c) else just use the first available
# -----------------------------
def pick_best_variant(entries):
    """
    entries: list of (full_id, seq) for the same base transcript id
    """
    # try -01 specifically
    for fid, seq in entries:
        if fid.endswith("-01"):
            return fid, seq

    # else try smallest numeric suffix if present
    def suffix_num(fid):
        m = re.search(r"-(\d+)$", fid)
        return int(m.group(1)) if m else 10**9

    entries_sorted = sorted(entries, key=lambda x: suffix_num(x[0]))
    return entries_sorted[0]

# -----------------------------
# 3) Create transcript_id from RAP ID
#    Example: Os02g0216300 -> Os02t0216300
# -----------------------------
df_new = df.copy()

df_new["transcript_id"] = (
    df_new["RAP ID"]
      .astype(str)
      .str.strip()
      .str.replace(r"g", "t", n=1, regex=True)   # only the first 'g' -> 't'
)

# -----------------------------
# 4) Map transcript_id to promoter sequence
# -----------------------------
def get_promoter_from_transcript(tid):
    if tid in base_to_entries:
        fid, seq = pick_best_variant(base_to_entries[tid])
        return seq
    return np.nan

df_new["promoter_sequence"] = df_new["transcript_id"].apply(get_promoter_from_transcript)

# -----------------------------
# 5) Quick sanity checks
# -----------------------------
matched = df_new["promoter_sequence"].notna().sum()
total = len(df_new)

print(f"Joined promoters: {matched:,}/{total:,} rows matched ({matched/total:.2%})")

print("\nExample matched rows:")
print(
    df_new.loc[df_new["promoter_sequence"].notna(), ["RAP ID", "transcript_id", "promoter_sequence"]]
      .head(5)
      .assign(promoter_sequence=lambda x: x["promoter_sequence"].str[:60] + "...")
)

print("\nExample unmatched rows:")
print(df_new.loc[df_new["promoter_sequence"].isna(), ["RAP ID", "transcript_id"]].head(5))


import os
output_path = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_3.csv"

# Make sure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# Save the dataframe
df_new.to_csv(output_path, index=False)
print(f" File saved successfully to:\n{output_path}")
print(f"Saved shape: {df_new.shape}")
