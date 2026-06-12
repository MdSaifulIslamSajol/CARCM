#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 06:27:28 2026

@author: saiful
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:29:57 2026

@author: saiful
"""

import pandas as pd

file_path = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_4.csv"

df = pd.read_csv(file_path)

print("Column names:")
print(df.columns.tolist())


has_duplicates = df.duplicated().any()
print("Are there any duplicate rows?", has_duplicates)

num_duplicates = df.duplicated().sum()
print("Number of duplicate rows:", num_duplicates)

duplicates = df[df.duplicated()]
print(duplicates)



df_unique = df.drop_duplicates()

print("Shape before removing duplicates:", df.shape)
print("Shape after removing duplicates:", df_unique.shape)

#%%
out_path = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_5.csv"

df_unique.to_csv(out_path, index=False)

print("Saved  dataset to:")
print(out_path)

