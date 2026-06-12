#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for per-stratum performance analysis.
======================================================
Loads predictions, trains a sequence-only ablation baseline,
and provides DeLong test + reporting helpers.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/saiful/CARCM/output"
CARCM_PRED = "/home/saiful/CARCM/models_carcm/best/best_predictions.csv"
CARCM_JSON = "/home/saiful/CARCM/models_carcm/best/best_results.json"
OUT_BASE = "/home/saiful/CARCM/per-stratum performance"

MIN_N_FOR_AUC = 50


# ── DeLong test (fast, Sun & Xu 2014) ───────────────────────────────────────

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, label_1_count):
    m = label_1_count
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_pvalue(y_true, prob_a, prob_b):
    """Two-sided DeLong p-value for AUC(a) vs AUC(b)."""
    y = np.asarray(y_true, dtype=float)
    order = (-y).argsort()
    m = int(y[order].sum())
    preds = np.vstack([np.asarray(prob_a)[order],
                       np.asarray(prob_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var_diff <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = float(diff / np.sqrt(var_diff))
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), p


# ── Data loading ─────────────────────────────────────────────────────────────

def load_test_data():
    """Return merged DataFrame with stratum columns + predictions."""
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    pred_df = pd.read_csv(CARCM_PRED)

    variety_cols = sorted([c for c in test_df.columns if c.startswith("Variety_")])
    tissue_cols = sorted([c for c in test_df.columns if c.startswith("Tissue_")])
    stage_cols = sorted([c for c in test_df.columns if c.startswith("Stage_")])

    test_df["variety"] = test_df[variety_cols].idxmax(axis=1).str.replace("Variety_", "", regex=False)
    test_df["tissue"] = test_df[tissue_cols].idxmax(axis=1).str.replace("Tissue_", "", regex=False)
    test_df["stage"] = test_df[stage_cols].idxmax(axis=1).str.replace("Stage_", "", regex=False)

    test_df["carcm_prob"] = pred_df["predicted_prob"].values
    test_df["true_label"] = pred_df["true_label"].astype(int).values

    return test_df, variety_cols, tissue_cols, stage_cols


def train_sequence_only_baseline(data_dir=DATA_DIR):
    """Train logistic regression on motif features only (no context)."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    with open(os.path.join(data_dir, "feature_groups.json")) as f:
        fg = json.load(f)

    seq_cols = (fg["binned"] + fg["relative_position"] + fg["first_position"]
                + fg["last_position"] + fg["density_gradient"]
                + fg["global_count"] + fg["cooccurrence"] + fg["seq_stats"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[seq_cols].values)
    y_train = train_df["label"].values
    X_test = scaler.transform(test_df[seq_cols].values)

    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                            random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    seq_probs = lr.predict_proba(X_test)[:, 1]

    return seq_probs


# ── Stratum analysis ─────────────────────────────────────────────────────────

def compute_stratum_table(df, stratum_col, seq_probs):
    """
    Compute per-stratum metrics.
    Returns DataFrame with one row per stratum value.
    """
    rows = []
    for name, grp in df.groupby(stratum_col):
        y = grp["true_label"].values
        carcm_p = grp["carcm_prob"].values
        seq_p = seq_probs[grp.index.values]

        n = len(y)
        n_up = int(y.sum())
        n_down = n - n_up
        ratio = round(n_up / n_down, 2) if n_down > 0 else float("inf")

        both_classes = len(np.unique(y)) == 2

        if both_classes and n >= MIN_N_FOR_AUC:
            carcm_auc = roc_auc_score(y, carcm_p)
            seq_auc = roc_auc_score(y, seq_p)
            _, _, delong_p = delong_pvalue(y, carcm_p, seq_p)
            delta = carcm_auc - seq_auc
        else:
            carcm_auc = np.nan
            seq_auc = np.nan
            delong_p = np.nan
            delta = np.nan

        carcm_acc = accuracy_score(y, (carcm_p >= 0.3).astype(int))
        seq_acc = accuracy_score(y, (seq_p >= 0.5).astype(int))

        rows.append({
            stratum_col: name,
            "n_test": n,
            "n_up": n_up,
            "n_down": n_down,
            "class_ratio": ratio,
            "carcm_auc": round(carcm_auc, 4) if not np.isnan(carcm_auc) else np.nan,
            "seq_only_auc": round(seq_auc, 4) if not np.isnan(seq_auc) else np.nan,
            "delta_auc": round(delta, 4) if not np.isnan(delta) else np.nan,
            "delong_p": delong_p,
            "carcm_acc": round(carcm_acc, 4),
            "seq_only_acc": round(seq_acc, 4),
        })

    result = pd.DataFrame(rows).sort_values("n_test", ascending=False).reset_index(drop=True)
    return result


def fmt_p(v):
    if pd.isna(v):
        return "\u2014"
    if v == 0 or v < 1e-300:
        return "< 1e-300"
    if v < 0.001:
        return f"{v:.2e}"
    return f"{v:.4f}"
