#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Validation Aggregation — combine results from all 5 folds.
================================================================
Reads per-fold JSON results + prediction CSVs, then produces:

  cross_validation/
      cv_per_fold_metrics.csv      — one row per fold
      cv_summary.csv               — mean ± std for each metric
      cv_concatenated_predictions.csv — all test predictions (full dataset)
      cv_full_dataset_metrics.json — metrics on concatenated predictions
      cv_fold_comparison.{png,pdf} — bar charts per metric per fold
      cv_radar_chart.{png,pdf}     — radar overlay of all folds vs mean

@author: saiful
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, average_precision_score,
)

warnings.filterwarnings("ignore")

CV_BASE = "/home/saiful/CARCM/benchmarks/cross_validation"
N_FOLDS = 5

plt.rcParams.update({
    'font.size': 12, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
})


def main():
    print("=" * 70)
    print("CROSS-VALIDATION — AGGREGATING RESULTS")
    print("=" * 70)

    # ── 1. Collect per-fold metrics from JSON ────────────────────────────────
    fold_metrics = []
    all_predictions = []

    for fold in range(N_FOLDS):
        rdir = os.path.join(CV_BASE, f"fold_{fold}", "models", f"fold_{fold}")
        json_path = os.path.join(rdir, f"fold_{fold}_results.json")
        pred_path = os.path.join(rdir, f"fold_{fold}_predictions.csv")

        if not os.path.exists(json_path):
            print(f"  WARNING: fold {fold} results not found — skipping")
            continue

        with open(json_path) as f:
            data = json.load(f)

        m = data['metrics']
        m['fold'] = fold
        fold_metrics.append(m)

        if os.path.exists(pred_path):
            preds = pd.read_csv(pred_path)
            preds['fold'] = fold
            all_predictions.append(preds)
        print(f"  Loaded fold {fold}: AUC={m['test_auc']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}")

    if not fold_metrics:
        print("\nNo fold results found — nothing to aggregate.")
        return

    df = pd.DataFrame(fold_metrics)

    # ── 2. Per-fold table ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PER-FOLD METRICS")
    print(f"{'=' * 70}")
    display_cols = ['fold', 'test_auc', 'accuracy', 'precision', 'recall',
                    'f1', 'mcc', 'hard_auc', 'easy_auc', 'optimal_threshold']
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    df.to_csv(os.path.join(CV_BASE, "cv_per_fold_metrics.csv"),
              index=False, float_format="%.4f")

    # ── 3. Mean ± std ────────────────────────────────────────────────────────
    metric_cols = [c for c in df.columns if c != 'fold']
    print(f"\n{'=' * 70}")
    print("MEAN ± STD ACROSS FOLDS")
    print(f"{'=' * 70}")

    rows = []
    for col in metric_cols:
        mean_v = df[col].mean()
        std_v = df[col].std()
        print(f"  {col:25s}: {mean_v:.4f} ± {std_v:.4f}")
        rows.append({'metric': col, 'mean': mean_v, 'std': std_v,
                     'formatted': f"{mean_v:.4f} ± {std_v:.4f}"})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(CV_BASE, "cv_summary.csv"),
                      index=False, float_format="%.6f")

    # ── 4. Concatenated predictions → full-dataset CV metrics ────────────────
    if all_predictions:
        full_pred = pd.concat(all_predictions, ignore_index=True)
        full_pred.to_csv(os.path.join(CV_BASE, "cv_concatenated_predictions.csv"),
                         index=False)

        y_true = full_pred['true_label'].values
        y_prob = full_pred['predicted_prob'].values
        y_pred = full_pred['predicted_label'].values

        full_metrics = {
            'auc': float(roc_auc_score(y_true, y_prob)),
            'pr_auc': float(average_precision_score(y_true, y_prob)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred)),
            'mcc': float(matthews_corrcoef(y_true, y_pred)),
            'n_samples': int(len(y_true)),
            'n_folds': int(len(all_predictions)),
        }
        with open(os.path.join(CV_BASE, "cv_full_dataset_metrics.json"), 'w') as f:
            json.dump(full_metrics, f, indent=2)

        print(f"\n{'=' * 70}")
        print("FULL-DATASET CV METRICS (concatenated test predictions)")
        print(f"{'=' * 70}")
        for k, v in full_metrics.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    plot_fold_comparison(df)
    plot_radar(df)

    print(f"\nAll results saved to {CV_BASE}/")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_fold_comparison(df):
    """Bar chart: each metric across folds with mean ± 1 SD band."""
    metrics = ['test_auc', 'f1', 'accuracy', 'mcc', 'precision', 'recall']
    available = [m for m in metrics if m in df.columns]
    n = len(available)
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 0.8, N_FOLDS))

    for i, metric in enumerate(available):
        ax = axes[i]
        folds = df['fold'].values
        vals = df[metric].values
        mean_v = vals.mean()
        std_v = vals.std()

        bars = ax.bar(folds, vals, color=colors[:len(folds)],
                       edgecolor='black', alpha=0.85)
        ax.axhline(y=mean_v, color='red', ls='--', lw=2,
                    label=f'Mean: {mean_v:.4f}')
        ax.fill_between([-0.6, max(folds) + 0.6],
                         mean_v - std_v, mean_v + std_v,
                         alpha=0.12, color='red',
                         label=f'±1 SD: {std_v:.4f}')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(folds)
        ax.legend(fontsize=8)
        ymin = max(0, vals.min() - 0.05)
        ymax = min(1, vals.max() + 0.05)
        ax.set_ylim([ymin, ymax])

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('5-Fold Cross-Validation: Per-Fold vs Average',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(CV_BASE, f"cv_fold_comparison.{ext}"))
    plt.close(fig)
    print("  Saved cv_fold_comparison.{png,pdf}")


def plot_radar(df):
    """Radar / spider chart: all folds + mean."""
    metrics = ['test_auc', 'f1', 'accuracy', 'mcc', 'precision', 'recall']
    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        return

    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    fold_colors = plt.cm.Set2(np.linspace(0, 0.8, N_FOLDS))
    for idx in range(len(df)):
        row = df.iloc[idx]
        vals = [row[m] for m in available] + [row[available[0]]]
        ax.plot(angles, vals, 'o-', color=fold_colors[idx],
                label=f'Fold {int(row["fold"])}', alpha=0.65, lw=1.5)

    mean_vals = [df[m].mean() for m in available] + [df[available[0]].mean()]
    ax.plot(angles, mean_vals, 's-', color='red', lw=3,
            label='Mean', alpha=0.9, markersize=7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n').title() for m in available],
                        fontsize=10)
    ax.set_title('Per-Fold Performance Radar', fontsize=14,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(CV_BASE, f"cv_radar_chart.{ext}"))
    plt.close(fig)
    print("  Saved cv_radar_chart.{png,pdf}")


if __name__ == "__main__":
    main()
