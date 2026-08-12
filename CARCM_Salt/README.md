# CARCM Salt-Stress Transfer Learning


## Directory layout

```
CARCM_Salt/
├── README.md                                  (this file)
├── drought_and_salt_feature_extraction_pipe_line2.png   (Figure: shared feature pipeline)
├── rice_positional_pipeline_salt.py           (47-motif salt dictionary + feature extraction)
├── data_preprocess/
│   ├── preprocess_salt_pipeline.py            (raw excel merge -> cleaned salt csv)
│   └── RiceMetaSys_Salt_merged_all_excels_6.csv   (final salt data, n = 3,232)
├── models_carcm_salt/
│   └── hpsearch_summary.csv                   (15-config salt hyperparameter search;
│                                                justifies lr=1e-4/hidden=128/dropout=0.5/batch=128
│                                                used below)
└── transfer_learning/
    ├── tl_features.py                         (builds the shared drought+salt feature space)
    ├── tl_transfer.py                         (trains scratch / full fine-tune / frozen conditions)
    ├── run_transfer_learning.sh               (runs both scripts end to end)
    ├── data/
    │   ├── drought_features.parquet           (44,502 drought rows, shared 685-col feature space)
    │   ├── salt_features.parquet              (3,232 salt rows, same feature space)
    │   ├── context_categories.json            (16 varieties, 8 tissues, 6 stages - union vocabulary)
    │   ├── shared_motifs.json                 (60-motif shared dictionary)
    │   ├── shared_feature_groups.json         (feature column groups, matches shared_motifs.json)
    │   └── top_cooccurrence_pairs.txt         (top-50 motif pairs, computed on drought only)
    └── results/
        ├── tl_comparison.csv                  (Table 12 numbers)
        ├── tl_comparison.json                 (same, with full precision)
        ├── tl_comparison.png                  (bar chart of the three conditions)
        └── tl_predictions_<condition>.csv     (per-sample salt-test predictions)
```

## Pipeline

1. **`data_preprocess/preprocess_salt_pipeline.py`** cleans the raw
   RiceMetaSys salt module (dedup, conflict removal, promoter retrieval)
   and writes `RiceMetaSys_Salt_merged_all_excels_6.csv`. This is the
   salt-side analogue of the drought pipeline's `_6.csv` file.

2. **`transfer_learning/tl_features.py`** builds the shared feature
   space used by everything downstream:
   - reads the salt csv above via `rice_positional_pipeline_salt.py`
     (one directory up from `transfer_learning/`)
   - reads the drought csv `../../data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv`
     (see path note below)
   - takes the union of the 54 drought motifs and 47 salt motifs (60
     shared motifs, MYB2 collision resolved in favor of the drought
     definition), and the union of variety/tissue/stage vocabularies
     (16 varieties, with IR64 recognized as shared between drought and
     salt)
   - writes `transfer_learning/data/drought_features.parquet` and
     `salt_features.parquet` in an identical 685-column layout

3. **`transfer_learning/tl_transfer.py`** trains and evaluates three
   conditions, all on the same salt 80/10/10 split, 5-seed ensembles:
   - `scratch` — CARCM trained on salt only
   - `transfer_full` — drought-pretrained encoder, fully fine-tuned on salt
   - `transfer_frozen` — drought-pretrained encoder frozen, only the
     output head trained on salt

   Results are written to `transfer_learning/results/tl_comparison.csv`
   — these are the exact numbers in Table 12.

## Path note before running

`tl_features.py` currently points `DROUGHT_6` at an external path. If
this folder is placed inside the main CARCM repository (alongside its
own `data_preprocess/`), update:

```python
DROUGHT_6 = "../data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
```

to match wherever `RiceMetaSys_Drought_merged_all_excels_6.csv` lives
relative to this folder in the repo. The file is identical to the one
used for the main drought results.

## Running from scratch

```bash
cd CARCM_Salt/transfer_learning
python tl_features.py                          # ~10 min, CPU
python tl_transfer.py --gpu 0 --num_seeds 5     # ~85 min, GPU
```

or simply:

```bash
bash run_transfer_learning.sh 0    # 0 = GPU index
```

## Hyperparameters

`hidden_dim=128, dropout=0.5, batch_size=128, learning_rate=1e-4`
(scratch/fine-tune) and `learning_rate=3e-4` (drought pretraining) were
selected by the search recorded in
`models_carcm_salt/hpsearch_summary.csv` — the winning configuration
(`combined_opt1`) reached test AUC 0.6686 on salt-only training,
compared with 0.6138 for the drought model's own learning rate
(1e-3), which is why the two models use different rates.
