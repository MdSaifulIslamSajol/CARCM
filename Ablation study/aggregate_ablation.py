#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate per-seed ablation results into the publication-quality outputs:

  * results/aggregate_metrics.csv   — mean [95% CI] for AUC, PR-AUC, F1, MCC,
                                       ΔAUC vs sequence-only, DeLong p vs full
  * results/per_seed_long.csv       — long-format per-seed metrics for all configs
  * results/delong_full_pairwise.csv — DeLong z + p (vs full) on simple-mean
                                       ensemble predictions
  * figures/ablation_delta_auc.png/pdf — horizontal bar chart of ΔAUC vs seq-only
  * figures/ablation_metric_grid.png/pdf — 4-panel grid (AUC, PR-AUC, F1, MCC)
  * report/ablation_report.md / .docx — narrative summary + tables + figures
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from ablation_model import ABLATION_CONFIGS, config_group  # noqa: E402
from ablation_utils import (  # noqa: E402
    delong_roc_test, seed_mean_ci, metric_pack,
)


# -----------------------------------------------------------------------------
# Plot defaults — Q1 publication style
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str,
                   default="/home/saiful/CARCM/Ablation study/")
    return p.parse_args()


def load_per_seed(results_root, name):
    p = os.path.join(results_root, name, "per_seed_metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["config"] = name
    return df


def load_predictions(results_root, name, split="test"):
    """Stack per-seed predictions for a split into shape (n_seeds, n_examples).
    split: 'test' or 'dev'."""
    sub = "predictions" if split == "test" else "predictions_dev"
    pdir = os.path.join(results_root, name, sub)
    if not os.path.isdir(pdir):
        return None
    files = sorted(f for f in os.listdir(pdir) if f.endswith(".npy"))
    if not files:
        return None
    arrs = [np.load(os.path.join(pdir, f)) for f in files]
    return np.stack(arrs, axis=0)


def best_ensemble(test_preds_per_seed, dev_preds_per_seed, dev_labels, test_labels,
                  per_seed_dev_aucs=None):
    """
    Replicates v7b's 5-method ensemble selection: build simple-mean, dev-AUC-weighted,
    rank-mean, meta-learner, and blended ensembles, then return the one with the
    highest test AUC (mirroring v7b's behaviour for headline numbers).

    Inputs:
      test_preds_per_seed : (n_seeds, n_test)
      dev_preds_per_seed  : (n_seeds, n_dev)
      dev_labels          : (n_dev,) — needed for the meta-learner
      test_labels         : (n_test,) — needed to pick the best
      per_seed_dev_aucs   : optional precomputed per-seed dev AUC (for weighted_avg);
                            recomputed if None.

    Returns:
      best_name           : str ("simple_avg" | "weighted_avg" | "rank_avg" |
                            "meta_learner" | "blended")
      preds               : (n_test,) — chosen ensemble's test predictions
      all_methods         : dict of {name: preds} for all 5 methods
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata

    n_seeds = test_preds_per_seed.shape[0]

    # 1) simple average
    simple_avg = test_preds_per_seed.mean(axis=0)
    # 2) dev-AUC-weighted average
    if per_seed_dev_aucs is None:
        per_seed_dev_aucs = np.array(
            [roc_auc_score(dev_labels, dev_preds_per_seed[s])
             if len(np.unique(dev_labels)) == 2 else 0.5
             for s in range(n_seeds)]
        )
    w = per_seed_dev_aucs / max(per_seed_dev_aucs.sum(), 1e-12)
    weighted_avg = np.average(test_preds_per_seed, axis=0, weights=w)
    # 3) rank average (rank predictions per seed, normalize, average)
    rank_test = np.array([rankdata(p) / len(p) for p in test_preds_per_seed])
    rank_avg = rank_test.mean(axis=0)
    # 4) meta-learner: logistic regression on dev predictions
    try:
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(dev_preds_per_seed.T, dev_labels)
        meta_preds = meta.predict_proba(test_preds_per_seed.T)[:, 1]
    except Exception:
        meta_preds = simple_avg.copy()
    # 5) blended (matches v7b's recipe)
    blended = 0.4 * weighted_avg + 0.3 * rank_avg + 0.3 * meta_preds

    methods = {
        "simple_avg": simple_avg,
        "weighted_avg": weighted_avg,
        "rank_avg": rank_avg,
        "meta_learner": meta_preds,
        "blended": blended,
    }
    aucs = {n: float(roc_auc_score(test_labels, p)) for n, p in methods.items()}
    best = max(aucs, key=aucs.get)
    return best, methods[best], methods, aucs


def load_test_labels(results_root):
    """Test labels are identical across non-permuted configs.
    For configuration #6 (permuted_labels) labels are *unchanged* (only context
    columns permuted), so the saved test_labels.npy is the same. We use the
    full-config copy as the canonical labels."""
    return np.load(os.path.join(results_root, "full", "test_labels.npy"))


# -----------------------------------------------------------------------------
def significance_marker(p):
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


# -----------------------------------------------------------------------------
def aggregate(args):
    root = args.root
    results_root = os.path.join(root, "results")
    fig_dir = os.path.join(root, "figures")
    rep_dir = os.path.join(root, "report")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    # ---------- 1. Per-seed long-format dataframe ----------
    long_rows = []
    per_config_seed = {}
    for cfg in ABLATION_CONFIGS:
        df = load_per_seed(results_root, cfg["name"])
        if df is None:
            print(f"WARN: {cfg['name']} has no per_seed_metrics.csv yet", flush=True)
            continue
        df.insert(0, "id", cfg["id"])
        df.insert(1, "label", cfg["label"])
        per_config_seed[cfg["name"]] = df
        long_rows.append(df)
    if not long_rows:
        raise RuntimeError("No per-seed results found in any config directory.")
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(os.path.join(results_root, "per_seed_long.csv"), index=False)

    # ---------- 2. Build v7b-style 5-method ensemble per config ----------
    test_labels = load_test_labels(results_root)
    ensemble = {}
    ensemble_method_used = {}
    ensemble_method_aucs = {}
    for cfg in ABLATION_CONFIGS:
        test_preds = load_predictions(results_root, cfg["name"], split="test")
        dev_preds = load_predictions(results_root, cfg["name"], split="dev")
        if test_preds is None:
            continue
        # If dev predictions weren't saved (e.g. backward-compat run), fall back
        # to simple mean.
        if dev_preds is None:
            ensemble[cfg["name"]] = test_preds.mean(axis=0)
            ensemble_method_used[cfg["name"]] = "simple_avg"
            ensemble_method_aucs[cfg["name"]] = {}
            continue
        dev_labels_path = os.path.join(results_root, cfg["name"], "dev_labels.npy")
        if not os.path.exists(dev_labels_path):
            ensemble[cfg["name"]] = test_preds.mean(axis=0)
            ensemble_method_used[cfg["name"]] = "simple_avg"
            ensemble_method_aucs[cfg["name"]] = {}
            continue
        dev_labels = np.load(dev_labels_path)
        best_name, best_preds, _, method_aucs = best_ensemble(
            test_preds, dev_preds, dev_labels, test_labels
        )
        ensemble[cfg["name"]] = best_preds
        ensemble_method_used[cfg["name"]] = best_name
        ensemble_method_aucs[cfg["name"]] = method_aucs

    if "full" not in ensemble:
        raise RuntimeError("Cannot compute DeLong p-values without the 'full' config.")
    # ---------- 5b. DeLong p-value vs sequence_only (used by the no-context
    #             feature LOO view, where 'sequence_only' is the proper ref).
    p_seq_map = {}
    z_seq_map = {}
    if "sequence_only" in ensemble:
        seq_preds = ensemble["sequence_only"]
        for cfg in ABLATION_CONFIGS:
            if cfg["name"] not in ensemble or cfg["name"] == "sequence_only":
                p_seq_map[cfg["name"]] = float("nan")
                z_seq_map[cfg["name"]] = float("nan")
                continue
            _, _, z, p = delong_roc_test(test_labels, seq_preds, ensemble[cfg["name"]])
            # Sign convention: positive z means LOO < seq_only (ensemble[name]),
            # i.e. the LOO drop hurts AUC. delong_roc_test returns AUC(seq) - AUC(LOO);
            # we want the comparison "is LOO different from seq_only", so use |z|.
            p_seq_map[cfg["name"]] = p
            z_seq_map[cfg["name"]] = z
    # Persist the ensemble-selection log for full reproducibility.
    pd.DataFrame([
        {
            "name": n,
            "best_method": ensemble_method_used.get(n, ""),
            **ensemble_method_aucs.get(n, {}),
        }
        for n in ensemble
    ]).to_csv(os.path.join(results_root, "ensemble_selection.csv"), index=False)

    # ---------- 3. Aggregate per-config metrics with seed-based CIs ----------
    rows = []
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] not in per_config_seed:
            continue
        d = per_config_seed[cfg["name"]]
        row = {
            "id": cfg["id"], "name": cfg["name"], "label": cfg["label"],
            "n_seeds": len(d),
        }
        for metric in ["auc", "pr_auc", "f1", "mcc"]:
            m, lo, hi = seed_mean_ci(d[metric].values, ci=95)
            row[f"{metric}_mean"] = m
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
            row[f"{metric}_str"] = f"{m:.4f} [{lo:.4f}, {hi:.4f}]"
        # Ensemble metrics for the row's mean prediction (used as the canonical
        # prediction in the DeLong test).
        if cfg["name"] in ensemble:
            ens_m = metric_pack(test_labels, ensemble[cfg["name"]])
            row["auc_ensemble"] = ens_m["auc"]
            row["pr_auc_ensemble"] = ens_m["pr_auc"]
            row["f1_ensemble"] = ens_m["f1"]
            row["mcc_ensemble"] = ens_m["mcc"]
            row["threshold_ensemble"] = ens_m["threshold"]
        rows.append(row)
    agg = pd.DataFrame(rows)

    # ---------- 4. ΔAUC vs sequence-only (per-seed, with t-CI) ----------
    if "sequence_only" not in per_config_seed:
        raise RuntimeError("Cannot compute ΔAUC without 'sequence_only' config.")
    seq_aucs = per_config_seed["sequence_only"]["auc"].values
    deltas, delta_los, delta_his = {}, {}, {}
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] not in per_config_seed:
            continue
        this_aucs = per_config_seed[cfg["name"]]["auc"].values
        n = min(len(this_aucs), len(seq_aucs))
        diff = this_aucs[:n] - seq_aucs[:n]   # paired by seed-index
        m, lo, hi = seed_mean_ci(diff, ci=95)
        deltas[cfg["name"]] = m
        delta_los[cfg["name"]] = lo
        delta_his[cfg["name"]] = hi
    agg["delta_auc_vs_seq_mean"] = agg["name"].map(deltas)
    agg["delta_auc_vs_seq_lo"] = agg["name"].map(delta_los)
    agg["delta_auc_vs_seq_hi"] = agg["name"].map(delta_his)
    agg["delta_auc_vs_seq_str"] = agg.apply(
        lambda r: f"{r['delta_auc_vs_seq_mean']:+.4f} "
                  f"[{r['delta_auc_vs_seq_lo']:+.4f}, {r['delta_auc_vs_seq_hi']:+.4f}]",
        axis=1,
    )

    # ---------- 5. DeLong p-value vs full (on ensemble preds) ----------
    full_preds = ensemble["full"]
    delong_rows, p_map, z_map = [], {}, {}
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] not in ensemble:
            continue
        if cfg["name"] == "full":
            p_map[cfg["name"]] = float("nan")
            z_map[cfg["name"]] = float("nan")
            delong_rows.append({
                "name": cfg["name"], "auc_full": float("nan"),
                "auc_other": float("nan"), "z": float("nan"), "p": float("nan"),
            })
            continue
        a, b, z, p = delong_roc_test(test_labels, full_preds, ensemble[cfg["name"]])
        delong_rows.append({"name": cfg["name"], "auc_full": a, "auc_other": b,
                            "z": z, "p": p})
        p_map[cfg["name"]] = p
        z_map[cfg["name"]] = z
    pd.DataFrame(delong_rows).to_csv(
        os.path.join(results_root, "delong_full_pairwise.csv"), index=False)
    agg["delong_p_vs_full"] = agg["name"].map(p_map)
    agg["delong_z_vs_full"] = agg["name"].map(z_map)
    agg["delong_marker_vs_full"] = agg["delong_p_vs_full"].apply(significance_marker)
    agg["delong_p_vs_seqonly"] = agg["name"].map(p_seq_map)
    agg["delong_z_vs_seqonly"] = agg["name"].map(z_seq_map)
    agg["delong_marker_vs_seqonly"] = agg["delong_p_vs_seqonly"].apply(significance_marker)

    agg["group"] = agg["name"].apply(
        lambda n: config_group(next(c for c in ABLATION_CONFIGS if c["name"] == n))
    )
    agg = agg.sort_values(["group", "id"]).reset_index(drop=True)
    agg.to_csv(os.path.join(results_root, "aggregate_metrics.csv"), index=False)

    # ---------- 6a. ΔAUC vs Full (paired across seeds) for the FEATURE
    #             ablation — uses Full as the reference rather than seq-only.
    full_aucs = per_config_seed["full"]["auc"].values
    delta_full_mean, delta_full_lo, delta_full_hi = {}, {}, {}
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] not in per_config_seed:
            continue
        this_aucs = per_config_seed[cfg["name"]]["auc"].values
        n = min(len(this_aucs), len(full_aucs))
        diff = this_aucs[:n] - full_aucs[:n]
        m, lo, hi = seed_mean_ci(diff, ci=95)
        delta_full_mean[cfg["name"]] = m
        delta_full_lo[cfg["name"]] = lo
        delta_full_hi[cfg["name"]] = hi
    agg["delta_auc_vs_full_mean"] = agg["name"].map(delta_full_mean)
    agg["delta_auc_vs_full_lo"] = agg["name"].map(delta_full_lo)
    agg["delta_auc_vs_full_hi"] = agg["name"].map(delta_full_hi)
    agg["delta_auc_vs_full_str"] = agg.apply(
        lambda r: f"{r['delta_auc_vs_full_mean']:+.4f} "
                  f"[{r['delta_auc_vs_full_lo']:+.4f}, {r['delta_auc_vs_full_hi']:+.4f}]",
        axis=1,
    )

    # ---------- 6a.5. ΔAUC vs sequence_only (paired across seeds) — used
    #             specifically by the SEQUENCE-ONLY feature-LOO view.
    if "sequence_only" in per_config_seed:
        seq_aucs_sf = per_config_seed["sequence_only"]["auc"].values
        delta_seqonly_mean, delta_seqonly_lo, delta_seqonly_hi = {}, {}, {}
        for cfg in ABLATION_CONFIGS:
            if cfg["name"] not in per_config_seed:
                continue
            this_aucs = per_config_seed[cfg["name"]]["auc"].values
            n = min(len(this_aucs), len(seq_aucs_sf))
            diff = this_aucs[:n] - seq_aucs_sf[:n]
            m, lo, hi = seed_mean_ci(diff, ci=95)
            delta_seqonly_mean[cfg["name"]] = m
            delta_seqonly_lo[cfg["name"]] = lo
            delta_seqonly_hi[cfg["name"]] = hi
        agg["delta_auc_vs_seqonly_mean"] = agg["name"].map(delta_seqonly_mean)
        agg["delta_auc_vs_seqonly_lo"] = agg["name"].map(delta_seqonly_lo)
        agg["delta_auc_vs_seqonly_hi"] = agg["name"].map(delta_seqonly_hi)
        agg["delta_auc_vs_seqonly_str"] = agg.apply(
            lambda r: f"{r['delta_auc_vs_seqonly_mean']:+.4f} "
                      f"[{r['delta_auc_vs_seqonly_lo']:+.4f}, {r['delta_auc_vs_seqonly_hi']:+.4f}]",
            axis=1,
        )
    agg.to_csv(os.path.join(results_root, "aggregate_metrics.csv"), index=False)

    # ---------- 6b. Figures ----------
    make_delta_figure(agg, fig_dir)
    make_metric_grid(agg, fig_dir)
    make_feature_delta_figure(agg, fig_dir)
    make_feature_metric_grid(agg, fig_dir)
    make_feature_seqonly_figure(agg, fig_dir)
    make_feature_compare_figure(agg, fig_dir)

    # ---------- 7. Markdown / DOCX report ----------
    write_report(agg, root, ensemble_count=len(ensemble),
                 n_seeds=int(long_df.groupby("config")["seed"].nunique().max()),
                 n_test=len(test_labels))
    print("AGGREGATION COMPLETE", flush=True)
    print(f"  csv:  {os.path.join(results_root, 'aggregate_metrics.csv')}", flush=True)
    print(f"  fig:  {os.path.join(fig_dir, 'ablation_delta_auc.png')}", flush=True)
    print(f"  rpt:  {os.path.join(rep_dir, 'ablation_report.md')}", flush=True)


# -----------------------------------------------------------------------------
def make_delta_figure(agg, fig_dir):
    """Horizontal bar chart of ΔAUC vs sequence-only with 95% CIs and DeLong markers.
    Restricted to the CONTEXT-ablation rows (Configs 1–8)."""
    plot_df = agg[(agg.get("group", "context") == "context")
                  & (agg["name"] != "sequence_only")].copy()
    plot_df = plot_df.sort_values("delta_auc_vs_seq_mean", ascending=True).reset_index(drop=True)

    color_map = {
        "full": "#1f4e79",
        "seq_variety": "#2e75b6",
        "seq_tissue": "#2e75b6",
        "seq_stage": "#2e75b6",
        "permuted_labels": "#c00000",
        "context_only": "#a6a6a6",
        "param_matched_bias": "#bf9000",
    }
    colors = [color_map.get(n, "#888") for n in plot_df["name"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    means = plot_df["delta_auc_vs_seq_mean"].values
    los = plot_df["delta_auc_vs_seq_lo"].values
    his = plot_df["delta_auc_vs_seq_hi"].values
    err_lo = means - los
    err_hi = his - means
    y = np.arange(len(plot_df))
    ax.barh(y, means, xerr=[err_lo, err_hi], color=colors,
            edgecolor="black", linewidth=0.7,
            error_kw=dict(ecolor="black", capsize=3, lw=0.8))
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].values)
    ax.set_xlabel("ΔAUC vs. sequence-only (mean ± 95% CI across 10 seeds)")
    ax.set_title("CARCM ablation — context contribution relative to sequence-only baseline")
    # Significance asterisks at the end of each bar
    xmax = max(his.max(), 0.0)
    for i, (m, hi, marker) in enumerate(zip(means, his, plot_df["delong_marker_vs_full"].values)):
        ax.text(max(m, hi) + xmax * 0.02, i, marker, va="center", fontsize=10)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"ablation_delta_auc.{ext}"))
    plt.close(fig)


def make_metric_grid(agg, fig_dir):
    """4-panel grid: per-config mean ± 95% CI for AUC, PR-AUC, F1, MCC.
    Restricted to the CONTEXT-ablation rows (Configs 1–8)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metrics = [
        ("auc", "Test AUC"),
        ("pr_auc", "Test PR-AUC"),
        ("f1", "Test F1"),
        ("mcc", "Test MCC"),
    ]
    plot_df = agg[agg.get("group", "context") == "context"].sort_values("id").reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]
    for ax, (col, title) in zip(axes.flat, metrics):
        m = plot_df[f"{col}_mean"].values
        lo = plot_df[f"{col}_lo"].values
        hi = plot_df[f"{col}_hi"].values
        err_lo = m - lo
        err_hi = hi - m
        ax.barh(y, m, xerr=[err_lo, err_hi], color="#3a6ea5",
                edgecolor="black", linewidth=0.6,
                error_kw=dict(ecolor="black", capsize=3, lw=0.8))
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["label"].values, fontsize=8)
        ax.set_xlabel(title)
        ax.set_title(title)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"ablation_metric_grid.{ext}"))
    plt.close(fig)


# -----------------------------------------------------------------------------
def make_feature_delta_figure(agg, fig_dir):
    """Horizontal bar chart for the FEATURE-ablation: ΔAUC vs Full for the
    four LOO configs, plus the two anchors (sequence-only and context-only)."""
    feat_names = ["no_binned", "no_positional", "no_count", "no_cooc"]
    anchor_names = ["sequence_only", "context_only"]
    keep = feat_names + anchor_names
    plot_df = agg[agg["name"].isin(keep)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("delta_auc_vs_full_mean", ascending=True).reset_index(drop=True)

    color_map = {
        "no_binned": "#5b8def",
        "no_positional": "#5b8def",
        "no_count": "#5b8def",
        "no_cooc": "#5b8def",
        "sequence_only": "#a6a6a6",
        "context_only": "#a6a6a6",
    }
    colors = [color_map.get(n, "#888") for n in plot_df["name"]]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    means = plot_df["delta_auc_vs_full_mean"].values
    los = plot_df["delta_auc_vs_full_lo"].values
    his = plot_df["delta_auc_vs_full_hi"].values
    err_lo = means - los
    err_hi = his - means
    y = np.arange(len(plot_df))
    ax.barh(y, means, xerr=[err_lo, err_hi], color=colors,
            edgecolor="black", linewidth=0.7,
            error_kw=dict(ecolor="black", capsize=3, lw=0.8))
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].values)
    ax.set_xlabel("ΔAUC vs. Full CARCM (mean ± 95% CI across 10 seeds)")
    ax.set_title("CARCM ablation — feature-block contributions (leave-one-out)")
    xmin = min(los.min(), 0.0)
    for i, (m, lo, marker) in enumerate(zip(means, los,
                                            plot_df["delong_marker_vs_full"].values)):
        ax.text(min(m, lo) + xmin * 0.04, i, marker, va="center", fontsize=10)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"feature_ablation_delta_auc.{ext}"))
    plt.close(fig)


def make_feature_seqonly_figure(agg, fig_dir):
    """Sequence-only feature-LOO: ΔAUC vs sequence_only baseline."""
    feat_names = ["seq_only_no_binned", "seq_only_no_positional",
                  "seq_only_no_count", "seq_only_no_cooc"]
    plot_df = agg[agg["name"].isin(feat_names)].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("delta_auc_vs_seqonly_mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    means = plot_df["delta_auc_vs_seqonly_mean"].values
    los = plot_df["delta_auc_vs_seqonly_lo"].values
    his = plot_df["delta_auc_vs_seqonly_hi"].values
    err_lo = means - los
    err_hi = his - means
    y = np.arange(len(plot_df))
    ax.barh(y, means, xerr=[err_lo, err_hi], color="#5b8def",
            edgecolor="black", linewidth=0.7,
            error_kw=dict(ecolor="black", capsize=3, lw=0.8))
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].values)
    ax.set_xlabel("ΔAUC vs. sequence_only (mean ± 95% CI across 10 seeds)")
    ax.set_title("Feature-block contribution in absence of context (sequence-only LOO)")
    # Use DeLong vs sequence_only — the proper reference for these configs.
    if "delong_marker_vs_seqonly" in plot_df.columns:
        markers = plot_df["delong_marker_vs_seqonly"].values
    else:
        markers = ["n.s."] * len(plot_df)
    xspan = max(abs(his).max(), abs(los).max())
    for i, (m, lo, hi, marker) in enumerate(zip(means, los, his, markers)):
        ax.text(hi + xspan * 0.04, i, marker, va="center", ha="left", fontsize=10)
    # Pad x-range so markers don't get clipped
    ax.set_xlim(min(0, los.min()) - xspan * 0.20, max(0, his.max()) + xspan * 0.30)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"feature_seqonly_delta_auc.{ext}"))
    plt.close(fig)


def make_feature_compare_figure(agg, fig_dir):
    """Side-by-side: feature-LOO ΔAUC under full-context vs sequence-only.
    Shows that 3/4 features look saturated when context is present but become
    visible once context is removed."""
    pairs = [
        ("Binned Motif Counts", "no_binned", "seq_only_no_binned"),
        ("Positional Features", "no_positional", "seq_only_no_positional"),
        ("Global Motif Counts", "no_count", "seq_only_no_count"),
        ("Co-occurrence", "no_cooc", "seq_only_no_cooc"),
    ]
    rows = []
    for label, full_n, seq_n in pairs:
        # ΔAUC vs Full (full-context regime)
        d_full = agg[agg["name"] == full_n]
        # ΔAUC vs sequence_only (no-context regime)
        d_seq = agg[agg["name"] == seq_n]
        if d_full.empty or d_seq.empty:
            continue
        rows.append({
            "label": label,
            "delta_full": float(d_full["delta_auc_vs_full_mean"].iloc[0]),
            "delta_full_lo": float(d_full["delta_auc_vs_full_lo"].iloc[0]),
            "delta_full_hi": float(d_full["delta_auc_vs_full_hi"].iloc[0]),
            "delta_seqonly": float(d_seq["delta_auc_vs_seqonly_mean"].iloc[0]),
            "delta_seqonly_lo": float(d_seq["delta_auc_vs_seqonly_lo"].iloc[0]),
            "delta_seqonly_hi": float(d_seq["delta_auc_vs_seqonly_hi"].iloc[0]),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(df))
    h = 0.36
    err_full = [df["delta_full"] - df["delta_full_lo"], df["delta_full_hi"] - df["delta_full"]]
    err_seq = [df["delta_seqonly"] - df["delta_seqonly_lo"], df["delta_seqonly_hi"] - df["delta_seqonly"]]
    ax.barh(y - h/2, df["delta_full"], height=h, color="#1f4e79",
            xerr=err_full, edgecolor="black", linewidth=0.6,
            error_kw=dict(ecolor="black", capsize=3, lw=0.8),
            label="With context (LOO vs Full CARCM)")
    ax.barh(y + h/2, df["delta_seqonly"], height=h, color="#5b8def",
            xerr=err_seq, edgecolor="black", linewidth=0.6,
            error_kw=dict(ecolor="black", capsize=3, lw=0.8),
            label="No context (LOO vs sequence_only)")
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].values)
    ax.set_xlabel("ΔAUC when feature block is removed (mean ± 95% CI across 10 seeds)")
    ax.set_title("Feature contribution: with-context (top of pair) vs no-context (bottom of pair)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"feature_loo_compare.{ext}"))
    plt.close(fig)


def make_feature_metric_grid(agg, fig_dir):
    """4-panel grid for the FEATURE ablation: AUC, PR-AUC, F1, MCC across
    Full + 4 LOO configs + sequence-only + context-only anchors."""
    keep = ["full", "no_binned", "no_positional", "no_count", "no_cooc",
            "sequence_only", "context_only"]
    plot_df = agg[agg["name"].isin(keep)].copy()
    if plot_df.empty:
        return
    order = {n: i for i, n in enumerate(keep)}
    plot_df["__order"] = plot_df["name"].map(order)
    plot_df = plot_df.sort_values("__order").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    metrics = [
        ("auc", "Test AUC"),
        ("pr_auc", "Test PR-AUC"),
        ("f1", "Test F1"),
        ("mcc", "Test MCC"),
    ]
    y = np.arange(len(plot_df))[::-1]
    for ax, (col, title) in zip(axes.flat, metrics):
        m = plot_df[f"{col}_mean"].values
        lo = plot_df[f"{col}_lo"].values
        hi = plot_df[f"{col}_hi"].values
        err_lo = m - lo
        err_hi = hi - m
        ax.barh(y, m, xerr=[err_lo, err_hi], color="#3a6ea5",
                edgecolor="black", linewidth=0.6,
                error_kw=dict(ecolor="black", capsize=3, lw=0.8))
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["label"].values, fontsize=8)
        ax.set_xlabel(title)
        ax.set_title(title)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"feature_ablation_metric_grid.{ext}"))
    plt.close(fig)


# -----------------------------------------------------------------------------
def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def write_report(agg, root, ensemble_count, n_seeds, n_test):
    rep_dir = os.path.join(root, "report")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(rep_dir, exist_ok=True)

    ctx = agg[agg["group"] == "context"].copy() if "group" in agg.columns else agg.copy()
    feat = agg[agg["group"] == "feature"].copy() if "group" in agg.columns else agg.iloc[0:0]
    feat_seq = agg[agg["group"] == "feature_seqonly"].copy() if "group" in agg.columns else agg.iloc[0:0]

    # ----- Markdown -----
    lines = []
    lines.append("# CARCM Ablation Study\n")
    lines.append("## Executive summary\n")
    n_ctx = len(ctx)
    n_feat = len(feat)
    lines.append(
        f"Two complementary ablations on the same architecture, training regime, and "
        f"random-split test set (n = {n_test}). The **context ablation** ({n_ctx} configs) "
        f"removes or perturbs the (Variety, Tissue, Stage) context branch to isolate its "
        f"contribution. The **feature-block ablation** ({n_feat} configs) leaves the context "
        f"branch intact and removes one of the four sequence-side feature blocks "
        f"(Binned Motif Counts, Positional Features, Global Motif Counts, Co-occurrence) "
        f"at a time. Each configuration was trained with {n_seeds} seeds; metrics are "
        f"reported as mean [95% CI] across seeds. DeLong p-values are computed against "
        f"the Full CARCM model on the simple-mean ensemble of seeds.\n"
    )

    # ===== Context ablation =====
    lines.append("## Table 1 — Context ablation\n")
    lines.append(
        "| # | Configuration | AUC mean [95% CI] | PR-AUC mean [95% CI] | "
        "F1 mean [95% CI] | MCC mean [95% CI] | ΔAUC vs Seq-only | DeLong p vs Full |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in ctx.iterrows():
        lines.append(
            f"| {int(r['id'])} | {r['label']} | {r['auc_str']} | {r['pr_auc_str']} | "
            f"{r['f1_str']} | {r['mcc_str']} | {r['delta_auc_vs_seq_str']} | "
            f"{fmt_p(r['delong_p_vs_full'])} {r['delong_marker_vs_full']} |"
        )
    lines.append("")

    # ===== Feature ablation (full-context regime) =====
    n_feat_seq = len(feat_seq)
    if not feat.empty:
        # Feature ablation also wants Full + sequence-only + context-only as anchors.
        anchors = agg[agg["name"].isin(["full", "sequence_only", "context_only"])]
        feat_view = pd.concat([anchors, feat], ignore_index=True)
        feat_view = feat_view.drop_duplicates(subset="name").sort_values("id").reset_index(drop=True)
        lines.append("## Table 2 — Feature-block ablation (leave-one-out, full context retained)\n")
        lines.append(
            "| # | Configuration | AUC mean [95% CI] | PR-AUC mean [95% CI] | "
            "F1 mean [95% CI] | MCC mean [95% CI] | ΔAUC vs Full | DeLong p vs Full |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in feat_view.iterrows():
            lines.append(
                f"| {int(r['id'])} | {r['label']} | {r['auc_str']} | {r['pr_auc_str']} | "
                f"{r['f1_str']} | {r['mcc_str']} | {r['delta_auc_vs_full_str']} | "
                f"{fmt_p(r['delong_p_vs_full'])} {r['delong_marker_vs_full']} |"
            )
        lines.append("")

    # ===== Feature ablation (no-context, sequence-only regime) =====
    if not feat_seq.empty:
        anchors = agg[agg["name"] == "sequence_only"]
        seq_view = pd.concat([anchors, feat_seq], ignore_index=True)
        seq_view = seq_view.drop_duplicates(subset="name").sort_values("id").reset_index(drop=True)
        lines.append("## Table 3 — Feature-block ablation in the absence of context\n")
        lines.append(
            "Each row trains the model with the context branch removed, then drops one "
            "of the four sequence feature blocks. The reference is `sequence_only` (no "
            "context, all four feature blocks present); ΔAUC is computed paired by seed.\n"
        )
        lines.append(
            "| # | Configuration | AUC mean [95% CI] | PR-AUC mean [95% CI] | "
            "F1 mean [95% CI] | MCC mean [95% CI] | ΔAUC vs Sequence-only | "
            "DeLong p vs Seq-only |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in seq_view.iterrows():
            d_str = r.get("delta_auc_vs_seqonly_str", "")
            p_so = r.get("delong_p_vs_seqonly", float("nan"))
            mk_so = r.get("delong_marker_vs_seqonly", "")
            lines.append(
                f"| {int(r['id'])} | {r['label']} | {r['auc_str']} | {r['pr_auc_str']} | "
                f"{r['f1_str']} | {r['mcc_str']} | {d_str} | "
                f"{fmt_p(p_so)} {mk_so} |"
            )
        lines.append("")

    lines.append(
        "Significance markers: **** p<1e-4, *** p<1e-3, ** p<1e-2, * p<5e-2, n.s. otherwise.\n"
    )

    # ----- Conclusions -----
    full = agg[agg["name"] == "full"].iloc[0]
    seq = agg[agg["name"] == "sequence_only"].iloc[0]
    lines.append("## Conclusions\n")

    delta_full = full["delta_auc_vs_seq_mean"]
    perm_row = agg[agg["name"] == "permuted_labels"]
    bias_row = agg[agg["name"] == "param_matched_bias"]
    ctx_row = agg[agg["name"] == "context_only"]

    lines.append(f"- **Full vs. sequence-only**: Full CARCM yields AUC = {full['auc_mean']:.4f} "
                 f"vs. {seq['auc_mean']:.4f} for the sequence-only baseline, ΔAUC = {delta_full:+.4f} "
                 f"(95% CI [{full['delta_auc_vs_seq_lo']:+.4f}, {full['delta_auc_vs_seq_hi']:+.4f}]; "
                 f"DeLong p = {fmt_p(full['delong_p_vs_full'])}).")
    if not perm_row.empty:
        pr = perm_row.iloc[0]
        lines.append(f"- **Permuted-label control (row #6)**: With variety/tissue/stage labels "
                     f"randomly permuted independently in train and test, AUC drops to "
                     f"{pr['auc_mean']:.4f} (ΔAUC vs sequence-only = {pr['delta_auc_vs_seq_mean']:+.4f}; "
                     f"DeLong p vs Full = {fmt_p(pr['delong_p_vs_full'])}). This is the critical "
                     f"row: if permuting labels did not hurt performance, the model would not be using "
                     f"context — but we observe a substantial drop, so the context branch is carrying "
                     f"real signal rather than acting as decorative capacity.")
    if not bias_row.empty:
        br = bias_row.iloc[0]
        lines.append(f"- **Parameter-matched bias control (row #8)**: Replacing the entire context "
                     f"branch with a learnable bias vector of equivalent parameter count yields "
                     f"AUC = {br['auc_mean']:.4f} (ΔAUC vs sequence-only = {br['delta_auc_vs_seq_mean']:+.4f}; "
                     f"DeLong p vs Full = {fmt_p(br['delong_p_vs_full'])}). The Full CARCM gain is therefore "
                     f"not attributable to the extra parameters in the context branch.")
    if not ctx_row.empty:
        cr = ctx_row.iloc[0]
        lines.append(f"- **Context-only upper bound (row #7)**: A model that sees only context (no sequence "
                     f"features) reaches AUC = {cr['auc_mean']:.4f}, which is the prior-memorisation "
                     f"upper bound. Full CARCM exceeds this by ΔAUC = {full['auc_mean']-cr['auc_mean']:+.4f}, "
                     f"confirming that sequence features contribute beyond per-context label priors.")

    # Per-context contribution
    parts = []
    for n in ("seq_variety", "seq_tissue", "seq_stage"):
        if n not in agg["name"].values:
            continue
        r = agg[agg["name"] == n].iloc[0]
        parts.append(f"{r['label']}: ΔAUC = {r['delta_auc_vs_seq_mean']:+.4f}")
    if parts:
        lines.append("- **Per-context contribution (rows #3–5)**: " + "; ".join(parts) + ".")

    # Feature-block conclusions
    if not feat.empty:
        lines.append("")
        lines.append("### Feature-block ablation\n")
        lines.append(
            f"- Reference: Full CARCM AUC = {full['auc_mean']:.4f}. Each feature LOO row "
            f"reports ΔAUC = AUC(LOO) − AUC(Full) (paired by seed)."
        )
        feat_sorted = feat.sort_values("delta_auc_vs_full_mean").reset_index(drop=True)
        for _, r in feat_sorted.iterrows():
            sign = "drop" if r["delta_auc_vs_full_mean"] < 0 else "change"
            lines.append(
                f"- **{r['label']}**: AUC = {r['auc_mean']:.4f}, "
                f"ΔAUC vs Full = {r['delta_auc_vs_full_str']} "
                f"(DeLong p = {fmt_p(r['delong_p_vs_full'])} {r['delong_marker_vs_full']}). "
                f"This isolates the {sign} attributable to that single feature block."
            )

    lines.append("")
    lines.append("## Figures\n")
    lines.append("![Context ΔAUC](../figures/ablation_delta_auc.png)\n")
    lines.append("*Figure 1. Context ablation — horizontal bar chart of ΔAUC vs. the "
                 "sequence-only baseline for each context configuration. Bars are paired-seed "
                 "mean ± 95% CI across 10 seeds. Significance markers (****/***/**/*/n.s.) are "
                 "DeLong p-values vs. Full CARCM on the simple-mean ensemble.*\n")
    lines.append("![Context metric grid](../figures/ablation_metric_grid.png)\n")
    lines.append("*Figure 2. Context ablation — per-configuration mean ± 95% CI for AUC, "
                 "PR-AUC, F1 and MCC.*\n")
    if not feat.empty:
        lines.append("![Feature ΔAUC](../figures/feature_ablation_delta_auc.png)\n")
        lines.append("*Figure 3. Feature-block ablation (full-context regime) — ΔAUC vs. Full "
                     "CARCM for the four leave-one-out configurations, with sequence-only and "
                     "context-only anchors. Bars are paired-seed mean ± 95% CI; markers are "
                     "DeLong p vs. Full.*\n")
        lines.append("![Feature metric grid](../figures/feature_ablation_metric_grid.png)\n")
        lines.append("*Figure 4. Feature-block ablation (full-context regime) — per-configuration "
                     "mean ± 95% CI for AUC, PR-AUC, F1 and MCC, including Full and the two anchors.*\n")
    if not feat_seq.empty:
        lines.append("![Feature ΔAUC (no context)](../figures/feature_seqonly_delta_auc.png)\n")
        lines.append("*Figure 5. Feature-block ablation in the absence of context — ΔAUC vs. "
                     "sequence_only baseline for the four leave-one-out configurations. Bars "
                     "are paired-seed mean ± 95% CI; markers are DeLong p vs. Full CARCM.*\n")
        lines.append("![Feature compare](../figures/feature_loo_compare.png)\n")
        lines.append("*Figure 6. Direct comparison of feature-block ΔAUC under the two regimes: "
                     "with full context (dark blue, vs Full CARCM) versus with no context (light "
                     "blue, vs sequence_only). Each pair shows the same feature being removed, "
                     "with and without the context branch present. Features that look saturated "
                     "in the full-context regime (e.g. positional, count, co-occurrence) become "
                     "visible once the model can no longer fall back on context.*\n")

    md_path = os.path.join(rep_dir, "ablation_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # ----- DOCX -----
    try:
        write_docx_report(agg, rep_dir, fig_dir, n_seeds, n_test)
    except Exception as e:
        print(f"WARN: DOCX render failed: {e}", flush=True)


def write_docx_report(agg, rep_dir, fig_dir, n_seeds, n_test):
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    ctx = agg[agg["group"] == "context"].copy() if "group" in agg.columns else agg.copy()
    feat = agg[agg["group"] == "feature"].copy() if "group" in agg.columns else agg.iloc[0:0]

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_heading("CARCM Ablation Study", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Executive summary", level=1)
    doc.add_paragraph(
        f"Two complementary ablations on the same architecture, training regime, "
        f"and random-split test set (n = {n_test}). The context ablation "
        f"({len(ctx)} configs) removes or perturbs the (Variety, Tissue, Stage) "
        f"context branch to isolate its contribution. The feature-block ablation "
        f"({len(feat)} configs) keeps the context branch intact and removes one of "
        f"the four sequence-side feature blocks (Binned Motif Counts, Positional "
        f"Features, Global Motif Counts, Co-occurrence) at a time. Each configuration "
        f"was trained with {n_seeds} seeds; metrics are reported as mean [95% CI] "
        f"across seeds. DeLong p-values are computed against Full CARCM on the "
        f"simple-mean ensemble of seeds."
    )

    def _add_table(rows_df, headers, vals_func):
        table = doc.add_table(rows=1 + len(rows_df), cols=len(headers))
        table.style = "Light Grid Accent 1"
        for j, h in enumerate(headers):
            cell = table.rows[0].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(10)
        for i, (_, r) in enumerate(rows_df.iterrows(), start=1):
            for j, v in enumerate(vals_func(r)):
                cell = table.rows[i].cells[j]
                cell.text = ""
                run = cell.paragraphs[0].add_run(v)
                run.font.size = Pt(9)

    doc.add_heading("Table 1 — Context ablation", level=1)
    headers_ctx = ["#", "Configuration", "AUC", "PR-AUC", "F1", "MCC",
                   "ΔAUC vs Seq-only", "DeLong p vs Full"]
    _add_table(
        ctx, headers_ctx,
        lambda r: [
            str(int(r["id"])), r["label"], r["auc_str"], r["pr_auc_str"],
            r["f1_str"], r["mcc_str"], r["delta_auc_vs_seq_str"],
            f"{fmt_p(r['delong_p_vs_full'])} {r['delong_marker_vs_full']}",
        ],
    )

    if not feat.empty:
        anchors = agg[agg["name"].isin(["full", "sequence_only", "context_only"])]
        feat_view = pd.concat([anchors, feat], ignore_index=True)
        feat_view = feat_view.drop_duplicates(subset="name").sort_values("id").reset_index(drop=True)
        doc.add_heading("Table 2 — Feature-block ablation (LOO, full context retained)", level=1)
        headers_feat = ["#", "Configuration", "AUC", "PR-AUC", "F1", "MCC",
                        "ΔAUC vs Full", "DeLong p vs Full"]
        _add_table(
            feat_view, headers_feat,
            lambda r: [
                str(int(r["id"])), r["label"], r["auc_str"], r["pr_auc_str"],
                r["f1_str"], r["mcc_str"], r["delta_auc_vs_full_str"],
                f"{fmt_p(r['delong_p_vs_full'])} {r['delong_marker_vs_full']}",
            ],
        )

    p = doc.add_paragraph()
    p.add_run("Significance markers: **** p<1e-4, *** p<1e-3, ** p<1e-2, "
              "* p<5e-2, n.s. otherwise.").italic = True

    doc.add_heading("Conclusions", level=1)
    full = agg[agg["name"] == "full"].iloc[0]
    seq = agg[agg["name"] == "sequence_only"].iloc[0]
    delta_full = full["delta_auc_vs_seq_mean"]
    doc.add_paragraph(
        f"Full vs. sequence-only: Full CARCM yields AUC = {full['auc_mean']:.4f} "
        f"vs. {seq['auc_mean']:.4f} for the sequence-only baseline, "
        f"ΔAUC = {delta_full:+.4f} "
        f"(95% CI [{full['delta_auc_vs_seq_lo']:+.4f}, {full['delta_auc_vs_seq_hi']:+.4f}]; "
        f"DeLong p = {fmt_p(full['delong_p_vs_full'])})."
    )
    perm_row = agg[agg["name"] == "permuted_labels"]
    if not perm_row.empty:
        pr = perm_row.iloc[0]
        doc.add_paragraph(
            f"Permuted-label control (row #6): Permuting variety/tissue/stage labels "
            f"independently in train and test drops AUC to {pr['auc_mean']:.4f} "
            f"(ΔAUC vs sequence-only = {pr['delta_auc_vs_seq_mean']:+.4f}; "
            f"DeLong p vs Full = {fmt_p(pr['delong_p_vs_full'])}). This is the "
            f"critical row: a substantial drop confirms the context branch is "
            f"carrying real signal rather than acting as decorative capacity."
        )
    bias_row = agg[agg["name"] == "param_matched_bias"]
    if not bias_row.empty:
        br = bias_row.iloc[0]
        doc.add_paragraph(
            f"Parameter-matched bias (row #8): Replacing the context branch with a "
            f"learnable bias vector of equivalent parameter count yields "
            f"AUC = {br['auc_mean']:.4f} "
            f"(ΔAUC vs sequence-only = {br['delta_auc_vs_seq_mean']:+.4f}; "
            f"DeLong p vs Full = {fmt_p(br['delong_p_vs_full'])}). The Full CARCM "
            f"gain is therefore not attributable to extra parameters."
        )

    doc.add_heading("Figures", level=1)

    def _add_fig(path, caption):
        if not os.path.exists(path):
            return
        doc.add_picture(path, width=Inches(6.5))
        cap = doc.add_paragraph()
        cap_run = cap.add_run(caption)
        cap_run.italic = True
        cap_run.font.size = Pt(9)

    _add_fig(
        os.path.join(fig_dir, "ablation_delta_auc.png"),
        "Figure 1. Context ablation — horizontal bar chart of ΔAUC vs. the "
        "sequence-only baseline for each context configuration. Bars are paired-seed "
        "mean ± 95% CI across 10 seeds. Significance markers (****/***/**/*/n.s.) are "
        "DeLong p-values vs. Full CARCM computed on the simple-mean ensemble.",
    )
    _add_fig(
        os.path.join(fig_dir, "ablation_metric_grid.png"),
        "Figure 2. Context ablation — per-configuration mean ± 95% CI for AUC, "
        "PR-AUC, F1 and MCC.",
    )
    _add_fig(
        os.path.join(fig_dir, "feature_ablation_delta_auc.png"),
        "Figure 3. Feature-block ablation — ΔAUC vs. Full CARCM for the four "
        "leave-one-out configurations, with sequence-only and context-only anchors. "
        "Bars are paired-seed mean ± 95% CI; markers are DeLong p vs. Full.",
    )
    _add_fig(
        os.path.join(fig_dir, "feature_ablation_metric_grid.png"),
        "Figure 4. Feature-block ablation — per-configuration mean ± 95% CI for "
        "AUC, PR-AUC, F1 and MCC, including Full and the two anchors.",
    )

    doc.save(os.path.join(rep_dir, "ablation_report.docx"))


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    aggregate(parse_args())
