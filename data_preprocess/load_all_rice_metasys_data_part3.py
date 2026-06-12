import pandas as pd
import os

# Path to your merged dataset
file_path = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_3.csv"

# Safety check
print("File exists?", os.path.exists(file_path))

# Load CSV
df_loaded = pd.read_csv(file_path)

# Basic info
print(" File loaded successfully!")
print("Shape:", df_loaded.shape)

print("\nColumns:")
print(df_loaded.columns.tolist())

print("\nFirst 5 rows:")
print(df_loaded.head())

print("\nMissing values per column:")
print(df_loaded.isna().sum())

#%%
# -----------------------------
# Drop rows with ANY NaN values
# -----------------------------
before_shape = df_loaded.shape

df_clean = df_loaded.dropna(how="any").reset_index(drop=True)

after_shape = df_clean.shape

print("\nDropped rows with any missing values.")
print("Before shape:", before_shape)
print("After shape :", after_shape)
print("Total rows dropped:", before_shape[0] - after_shape[0])

# Verify no missing values remain
print("\nMissing values after cleaning:")
print(df_clean.isna().sum())

# Preview cleaned data
print("\nFirst 5 rows of cleaned dataframe:")
print(df_clean.head())
# %%Create a new dataframe with selected columns (in order)
# Define the exact column order you want
selected_cols = [
    "promoter_sequence",
    "Variety",
    "Tissue",
    "Stage",
    "Gene Regulation"
]

# Create the new dataframe
df_model = df_clean[selected_cols].copy()

# Sanity checks
print("New modeling dataframe created!")
print("Shape:", df_model.shape)

print("\nColumns (in order):")
print(df_model.columns.tolist())

print("\nFirst 5 rows:")
print(df_model.head())

print("\nMissing values check:")
print(df_model.isna().sum())

#%%
# Count the distribution of Gene Regulation labels
label_counts = df_clean["Gene Regulation"].value_counts()

print("Gene Regulation label distribution:")
print(label_counts)

# Optional: more readable format
num_up = label_counts.get("Up Regulated", 0)
num_down = label_counts.get("Down Regulated", 0)

print("\nSummary:")
print("Up Regulated   :", num_up)
print("Down Regulated :", num_down)
print("Total samples  :", num_up + num_down)

#%%
import os

output_path = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_4.csv"

# Make sure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save df_model
df_model.to_csv(output_path, index=False)

print("Saved df_model successfully.")
print("Path:", output_path)
print("Saved shape:", df_model.shape)


