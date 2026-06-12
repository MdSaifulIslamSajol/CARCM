#!/usr/bin/env python3
"""
Common utilities for all benchmark models.

Loads the exact same feature matrix that rice_classifier_v7.py uses:
  - positional-binned motif counts
  - relative position, first/last position, density gradient
  - global motif counts, co-occurrence pairs, seq stats
  - one-hot Variety / Tissue / Stage context
All features concatenated into one tabular matrix X; label -> y.

Also replicates V7's hard/easy example identification so every benchmark
reports hard_auc and easy_auc on the same definition (identical motif-count
fingerprints with conflicting labels in training -> 'hard').

@author: saiful
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef, f1_score,
    average_precision_score, roc_curve, precision_recall_curve,
)

DATA_DIR = "/home/saiful/CARCM/output/"
RESULTS_DIR = "/home/saiful/CARCM/benchmarks/results"
CARCM_RESULTS = "/home/saiful/CARCM/models_carcm/best/best_results.json"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_feature_groups():
    with open(os.path.join(DATA_DIR, "feature_groups.json")) as f:
        return json.load(f)


def _feature_column_order(fg):
    # Matches the feature ordering produced by the positional pipeline; context
    # one-hot columns are appended last.
    numeric = (
        fg["binned"] + fg["relative_position"] + fg["first_position"]
        + fg["last_position"] + fg["density_gradient"] + fg["global_count"]
        + fg["cooccurrence"] + fg["seq_stats"]
    )
    context = fg["variety"] + fg["tissue"] + fg["stage"]
    return numeric, context


def load_data(scale="standard"):
    """Load train/dev/test splits.

    Returns a dict with X/y matrices, feature names, and aux info.

    scale: 'standard' => StandardScaler fit on numeric features (context kept
           one-hot). 'none' => raw values.
    """
    fg = load_feature_groups()
    numeric_cols, context_cols = _feature_column_order(fg)
    feat_cols = numeric_cols + context_cols

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Hard/easy classification replicates V7 logic via fingerprint on the
    # global motif count vector.
    count_cols = fg["global_count"]

    def fingerprint(df):
        arr = np.round(df[count_cols].values, 4)
        return np.array([hash(tuple(row)) for row in arr])

    train_fp = fingerprint(train_df)
    test_fp = fingerprint(test_df)
    tmp = train_df[["label"]].copy()
    tmp["fp"] = train_fp
    conflicting = tmp.groupby("fp")["label"].nunique()
    hard_set = set(conflicting[conflicting > 1].index)
    test_hard_mask = np.isin(test_fp, list(hard_set))

    X_train = train_df[feat_cols].values.astype(np.float32)
    X_dev = dev_df[feat_cols].values.astype(np.float32)
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    y_dev = dev_df["label"].values.astype(np.int64)
    y_test = test_df["label"].values.astype(np.int64)

    scaler = None
    if scale == "standard":
        n_num = len(numeric_cols)
        scaler = StandardScaler()
        X_train[:, :n_num] = scaler.fit_transform(X_train[:, :n_num])
        X_dev[:, :n_num] = scaler.transform(X_dev[:, :n_num])
        X_test[:, :n_num] = scaler.transform(X_test[:, :n_num])

    return dict(
        X_train=X_train, y_train=y_train,
        X_dev=X_dev, y_dev=y_dev,
        X_test=X_test, y_test=y_test,
        feat_cols=feat_cols,
        numeric_cols=numeric_cols,
        context_cols=context_cols,
        test_hard_mask=test_hard_mask,
        scaler=scaler,
    )


def find_optimal_threshold(y_true, y_prob):
    """Pick threshold that maximizes F1 (V7's convention)."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def compute_metrics(y_true, y_prob, hard_mask, threshold=None, model_name=""):
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    hard_auc = easy_auc = float("nan")
    if hard_mask.sum() > 0 and len(np.unique(y_true[hard_mask])) > 1:
        hard_auc = roc_auc_score(y_true[hard_mask], y_prob[hard_mask])
    easy_mask = ~hard_mask
    if easy_mask.sum() > 0 and len(np.unique(y_true[easy_mask])) > 1:
        easy_auc = roc_auc_score(y_true[easy_mask], y_prob[easy_mask])

    return {
        "model": model_name,
        "test_auc": float(auc),
        "test_ap": float(ap),
        "hard_auc": float(hard_auc),
        "easy_auc": float(easy_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "optimal_threshold": float(threshold),
        "n_test": int(len(y_true)),
        "n_hard": int(hard_mask.sum()),
    }


def save_result(result, y_true, y_prob, hard_mask):
    name = result["model"]
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "is_hard": hard_mask.astype(int),
    }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    print(f"[{name}] AUC={result['test_auc']:.4f}  F1={result['f1']:.4f}  "
          f"MCC={result['mcc']:.4f}  hard={result['hard_auc']:.4f}  "
          f"easy={result['easy_auc']:.4f}")
