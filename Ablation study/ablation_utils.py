#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for the ablation study: DeLong test, bootstrap CIs, metric helpers.
"""

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef,
)


# =============================================================================
# DeLong's test for paired ROC AUC differences
# =============================================================================

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


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong (Sun & Xu 2014) for K paired models on the same labels."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    K = predictions_sorted_transposed.shape[0]

    tx = np.empty([K, m], dtype=float)
    ty = np.empty([K, n], dtype=float)
    tz = np.empty([K, m + n], dtype=float)
    for r in range(K):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(labels, preds_a, preds_b):
    """
    Compute DeLong's two-sided p-value for AUC(A) - AUC(B) on paired samples.
    Returns (auc_a, auc_b, z_score, p_value).
    """
    labels = np.asarray(labels).astype(int)
    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)
    assert labels.shape == preds_a.shape == preds_b.shape

    order = (-labels).argsort(kind="mergesort")
    label_1_count = int(labels.sum())
    preds_sorted = np.vstack([preds_a, preds_b])[:, order]
    aucs, cov = _fast_delong(preds_sorted, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0 or not np.isfinite(var):
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    z = diff / np.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# =============================================================================
# Bootstrap 95% CI for any metric callable on (labels, preds)
# =============================================================================

def bootstrap_ci(labels, preds, metric_fn, n_boot=1000, seed=42, ci=95):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            vals.append(metric_fn(labels[idx], preds[idx]))
        except Exception:
            continue
    vals = np.asarray(vals)
    lo = float(np.percentile(vals, (100 - ci) / 2))
    hi = float(np.percentile(vals, 100 - (100 - ci) / 2))
    return lo, hi


# =============================================================================
# Across-seed mean ± normal-approx 95% CI
# =============================================================================

def seed_mean_ci(values, ci=95):
    """Return (mean, lo, hi) using a t-based 95% CI for a small sample of seeds."""
    from scipy.stats import t as t_dist
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    m = arr.mean()
    s = arr.std(ddof=1)
    df = len(arr) - 1
    crit = t_dist.ppf(0.5 + ci / 200.0, df)
    se = s / np.sqrt(len(arr))
    return float(m), float(m - crit * se), float(m + crit * se)


# =============================================================================
# Per-prediction metric pack (used for both per-seed and ensemble)
# =============================================================================

def metric_pack(labels, preds, threshold=None, dev_labels=None, dev_preds=None):
    """
    Returns dict with auc, pr_auc, f1, mcc.

    f1 and mcc are computed at a decision threshold that is tuned on the
    DEVELOPMENT split (`dev_labels` / `dev_preds`) by scanning [0.3, 0.7] for the
    F1 optimum, then applied to `labels` / `preds`. Tuning on the evaluation
    split itself would pick the operating point using the same data the metrics
    are reported on. An explicit `threshold` overrides tuning; if no development
    predictions are supplied the threshold falls back to 0.5.
    """
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds, dtype=float)
    auc = float(roc_auc_score(labels, preds))
    ap = float(average_precision_score(labels, preds))
    if threshold is None:
        if dev_labels is not None and dev_preds is not None:
            dl = np.asarray(dev_labels).astype(int)
            dp = np.asarray(dev_preds, dtype=float)
            best_t, best_f1 = 0.5, -1.0
            for t in np.arange(0.30, 0.71, 0.01):
                f1 = f1_score(dl, (dp >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            threshold = best_t
        else:
            threshold = 0.5
    yhat = (preds >= threshold).astype(int)
    f1 = float(f1_score(labels, yhat, zero_division=0))
    mcc = float(matthews_corrcoef(labels, yhat))
    return {"auc": auc, "pr_auc": ap, "f1": f1, "mcc": mcc, "threshold": float(threshold)}
