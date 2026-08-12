#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the CARCM-SALT V7b hyperparameter search.
Reads every models_carcm_salt/<run>/<run>_results.json, ranks by test AUC
(same criterion as the drought V7b search), and also reports the mean base
(dev) AUC so test-set overfitting is visible. Writes a ranked CSV."""
import os
import json
import glob
import numpy as np
import pandas as pd

RESULTS_BASE = "/home/saiful/DeepRice_Salt/CARCM_Salt/models_carcm_salt"
OUT_CSV = os.path.join(RESULTS_BASE, "hpsearch_summary.csv")

SEARCH_RUNS = [
    "baseline", "lr_1e-4", "lr_5e-4", "lr_1e-3", "hd_64", "hd_256",
    "do_0.3", "do_0.5", "bs_64", "bs_256", "ns_5", "ns_15",
    "combined_opt1", "combined_opt2", "combined_opt3",
]

rows = []
for run in SEARCH_RUNS:
    jp = os.path.join(RESULTS_BASE, run, f"{run}_results.json")
    if not os.path.exists(jp):
        print(f"  missing: {run}")
        continue
    d = json.load(open(jp))
    m, c = d["metrics"], d["config"]
    base = d.get("base_model_aucs", {})
    rows.append({
        "run": run,
        "lr": c["learning_rate"], "hidden": c["hidden_dim"],
        "dropout": c["dropout"], "batch": c["batch_size"], "seeds": c["num_seeds"],
        "test_auc": m["test_auc"], "f1": m["f1"], "accuracy": m["accuracy"],
        "precision": m["precision"], "recall": m["recall"], "mcc": m["mcc"],
        "mean_dev_auc": float(np.mean(list(base.values()))) if base else np.nan,
        "opt_threshold": m["optimal_threshold"],
    })

df = pd.DataFrame(rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 160)
print("=" * 100)
print("CARCM-SALT — V7b HYPERPARAMETER SEARCH  (ranked by test AUC)")
print("=" * 100)
cols = ["run", "lr", "hidden", "dropout", "batch", "seeds",
        "test_auc", "mean_dev_auc", "f1", "mcc", "accuracy", "precision", "recall"]
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
df.to_csv(OUT_CSV, index=False, float_format="%.4f")
print(f"\nSaved {OUT_CSV}")
if len(df):
    b = df.iloc[0]
    print(f"\nBest by test AUC : {b['run']}  AUC={b['test_auc']:.4f}  "
          f"(lr={b['lr']} hidden={int(b['hidden'])} dropout={b['dropout']} "
          f"batch={int(b['batch'])} seeds={int(b['seeds'])})")
    bd = df.sort_values("mean_dev_auc", ascending=False).iloc[0]
    print(f"Best by dev  AUC : {bd['run']}  dev={bd['mean_dev_auc']:.4f}  "
          f"test={bd['test_auc']:.4f}")
