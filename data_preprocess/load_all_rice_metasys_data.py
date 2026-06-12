# -*- coding: utf-8 -*-

import pandas as pd

# Load the CSV file
file_path = "/home/saiful/DeepRice/RiceMetaSys Drought RGs/All data/RiceMetaSys_Drought_merged_all_excels_.csv"

# Try reading the CSV file
df = pd.read_csv(file_path)

print("File loaded successfully!")
print(f"Shape of the dataset: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nData types:")
print(df.dtypes)
#%%
# Create new DataFrame with reordered columns
new_df = df[['RAP ID', 'Tissue', 'Stage', 'Variety', 'Gene Regulation']].copy()

# Display the shape and first few rows of the new DataFrame
print(f"New DataFrame shape: {new_df.shape}")
print(f"\nFirst few rows of new DataFrame:")
print(new_df.head())

print(f"\nColumn names in new DataFrame (Gene Regulation is now last):")
print(new_df.columns.tolist())

print(f"\nData types in new DataFrame:")
print(new_df.dtypes)

# Check for missing values
print(f"\nMissing values in new DataFrame:")
print(new_df.isnull().sum())
#%%
# Count rows before dropping
print(f"Total rows before dropping NaN: {len(new_df)}")

# Drop all rows with any NaN values in any column
clean_df = new_df.dropna()

# Count rows after dropping
print(f"Total rows after dropping NaN: {len(clean_df)}")
print(f"Rows dropped: {len(new_df) - len(clean_df)}")

# Check if there are any NaN values left
print(f"\nMissing values in clean DataFrame:")
print(clean_df.isnull().sum())

# Display first few rows of cleaned DataFrame
print(f"\nFirst few rows of cleaned DataFrame:")
print(clean_df.head())

print(f"\nShape of cleaned DataFrame: {clean_df.shape}")

#%%
# Print unique values for each column
print("Unique Tissue values:")
print(sorted(clean_df['Tissue'].unique()))
print(f"Count: {clean_df['Tissue'].nunique()}\n")

print("Unique Stage values:")
print(sorted(clean_df['Stage'].unique()))
print(f"Count: {clean_df['Stage'].nunique()}\n")

print("Unique Variety values:")
print(sorted(clean_df['Variety'].unique()))
print(f"Count: {clean_df['Variety'].nunique()}\n")

#%%
# Create a copy to preserve original if needed
corrected_df = clean_df.copy()

# Create a mapping dictionary for corrections
corrections = {
    'Anth1er': 'Anther',
    'FlagLeaf': 'Flag Leaf',  # or 'Flag Leaf' -> 'FlagLeaf', choose your preferred format
    'Leaf1': 'Leaf',
    'Root1': 'Root'
}

# Apply the corrections to the Tissue column
corrected_df['Tissue'] = corrected_df['Tissue'].replace(corrections)

# Verify the corrections
print("Corrected unique Tissue values:")
print(sorted(corrected_df['Tissue'].unique()))
print(f"Count: {corrected_df['Tissue'].nunique()}\n")

print("Tissue value counts after correction:")
print(corrected_df['Tissue'].value_counts())

# Show sample rows to confirm corrections
print("\nSample rows after correction:")
print(corrected_df.head(10))

#%%
# Continue with the corrected_df from previous step
# Create a mapping for stage corrections
stage_corrections = {
    'VegetativeStage': 'Vegetative Stage'
}

# Apply the corrections to the Stage column
corrected_df['Stage'] = corrected_df['Stage'].replace(stage_corrections)

# Verify all unique Stage values after correction
print("Corrected unique Stage values:")
print(sorted(corrected_df['Stage'].unique()))
print(f"Count: {corrected_df['Stage'].nunique()}\n")

print("Stage value counts after correction:")
print(corrected_df['Stage'].value_counts())

# Optional: Show the complete summary of all corrected columns
print("\n=== FINAL SUMMARY ===")
print("\nCorrected unique Tissue values:")
print(sorted(corrected_df['Tissue'].unique()))
print(f"Count: {corrected_df['Tissue'].nunique()}")

print("\nCorrected unique Stage values:")
print(sorted(corrected_df['Stage'].unique()))
print(f"Count: {corrected_df['Stage'].nunique()}")

print("\nUnique Variety values:")
print(sorted(corrected_df['Variety'].unique()))
print(f"Count: {corrected_df['Variety'].nunique()}")

print("\nUnique Gene Regulation values:")
print(sorted(corrected_df['Gene Regulation'].unique()))
print(f"Count: {corrected_df['Gene Regulation'].nunique()}")
#%%

import os

# Save DataFrame
output_path = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_2.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
corrected_df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(f"Shape: {corrected_df.shape}")
print(f"Rows: {len(corrected_df)}, Columns: {len(corrected_df.columns)}")