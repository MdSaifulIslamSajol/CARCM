#!/usr/bin/env python3
"""Aggregate benchmark results + CARCM (CARCM lr_1e-3) into Q1-journal artifacts.

Outputs in benchmarks/results/summary/:
  - benchmark_comparison.csv           (all metrics per model)
  - benchmark_comparison.tex           (LaTeX booktabs table)
  - metrics_bar.{png,pdf}              (bar chart of AUC/F1/MCC)
  - radar_comparison.{png,pdf}         (radar plot of 6 metrics)
  - roc_curves.{png,pdf}               (overlaid ROC)
  - pr_curves.{png,pdf}                (overlaid precision-recall)
  - hard_vs_easy.{png,pdf}             (hard-AUC vs easy-AUC grouped bars)
  - calibration.{png,pdf}              (reliability diagrams)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve

RESULTS_DIR = "/home/saiful/CARCM/benchmarks/results"
CARCM_JSON = "/home/saiful/CARCM/models_carcm/best/best_results.json"
CARCM_PRED = "/home/saiful/CARCM/models_carcm/best/best_predictions.csv"
OUT_DIR = os.path.join(RESULTS_DIR, "summary")
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

# Non-red palette for baseline models (no red shades)
_BASELINE_COLORS = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf",
    "#7f7f7f", "#bcbd22", "#ff7f0e", "#377eb8", "#4daf4a",
    "#984ea3", "#a65628", "#66c2a5",
]
CARCM_COLOR = "#e63946"
CARCM_LW = 2.2
BASELINE_LW = 1.2

METRIC_COLS = [
    "test_auc", "test_ap", "hard_auc", "easy_auc",
    "accuracy", "precision", "recall", "f1", "mcc",
]


def load_benchmark_results():
    rows, preds = [], {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        mpath = os.path.join(RESULTS_DIR, name, "metrics.json")
        ppath = os.path.join(RESULTS_DIR, name, "predictions.csv")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            m = json.load(f)
        rows.append(m)
        if os.path.exists(ppath):
            preds[m["model"]] = pd.read_csv(ppath)
    return rows, preds


def load_CARCM():
    with open(CARCM_JSON) as f:
        d = json.load(f)
    m = d["metrics"]
    row = {
        "model": "CARCM",
        "test_auc": m["test_auc"],
        "test_ap": float("nan"),
        "hard_auc": m["hard_auc"],
        "easy_auc": m["easy_auc"],
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "mcc": m["mcc"],
        "optimal_threshold": m["optimal_threshold"],
    }
    pred_df = None
    if os.path.exists(CARCM_PRED):
        raw = pd.read_csv(CARCM_PRED)
        rename = {"true_label": "y_true", "predicted_prob": "y_prob",
                  "y_true": "y_true", "y_prob": "y_prob"}
        pred_df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
        from sklearn.metrics import average_precision_score
        row["test_ap"] = float(
            average_precision_score(pred_df["y_true"], pred_df["y_prob"])
        )
    return row, pred_df


def build_table(rows):
    df = pd.DataFrame(rows)
    # Order: ours at bottom for emphasis
    priority = [
        "GaussianNB", "BernoulliNB", "KNN", "DecisionTree",
        "LogisticRegression", "SVM_RBF", "RandomForest",
        "XGBoost", "LightGBM", "CatBoost", "TabNet", "FTTransformer",
        "CARCM",
    ]
    df["__ord"] = df["model"].apply(lambda x: priority.index(x) if x in priority else 99)
    df = df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    keep = ["model"] + METRIC_COLS + (["optimal_threshold"] if "optimal_threshold" in df.columns else [])
    df = df[[c for c in keep if c in df.columns]]
    return df


def write_table(df):
    csv_path = os.path.join(OUT_DIR, "benchmark_comparison.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"wrote {csv_path}")

    # LaTeX
    disp = df.copy()
    for c in METRIC_COLS:
        if c in disp.columns:
            disp[c] = disp[c].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "--")
    if "optimal_threshold" in disp.columns:
        disp["optimal_threshold"] = disp["optimal_threshold"].map(
            lambda v: f"{v:.2f}" if pd.notnull(v) else "--"
        )
    tex = disp.to_latex(index=False, escape=False,
                        column_format="l" + "r" * (disp.shape[1] - 1))
    tex = "% Requires \\usepackage{booktabs}\n" + tex
    with open(os.path.join(OUT_DIR, "benchmark_comparison.tex"), "w") as f:
        f.write(tex)
    print("wrote benchmark_comparison.tex")


def plot_bars(df):
    metrics = ["test_auc", "f1", "mcc", "accuracy"]
    titles = ["Test AUC", "F1", "MCC", "Accuracy"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    palette = [CARCM_COLOR if m == "CARCM" else "#8ecae6" for m in df["model"]]
    for ax, col, title in zip(axes, metrics, titles):
        vals = df[col].values
        bars = ax.bar(df["model"], vals, color=palette, edgecolor="black", linewidth=0.4)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(1.0, vals.max() * 1.05))
        ax.tick_params(axis="x", rotation=35)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Same-feature benchmark comparison on held-out test set",
                 y=1.03, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"metrics_bar.{ext}"))
    plt.close(fig)


def plot_radar(df):
    metrics = ["test_auc", "hard_auc", "easy_auc", "f1", "mcc", "accuracy"]
    labels = ["Test\nAUC", "Hard\nAUC", "Easy\nAUC", "F1", "MCC", "Accuracy"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    bi = 0
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] if pd.notnull(row[m]) else 0 for m in metrics]
        vals += vals[:1]
        is_carcm = row["model"] == "CARCM"
        color = CARCM_COLOR if is_carcm else _BASELINE_COLORS[bi % len(_BASELINE_COLORS)]
        lw = CARCM_LW if is_carcm else 1.4
        if not is_carcm:
            bi += 1
        ax.plot(angles, vals, linewidth=lw, color=color, label=row["model"])
        ax.fill(angles, vals, alpha=0.18 if is_carcm else 0.05, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Multi-metric comparison (higher is better)", y=1.08,
                 fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"radar_comparison.{ext}"))
    plt.close(fig)


def plot_roc_pr(preds):
    # ROC
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bi = 0
    for i, (name, df) in enumerate(preds.items()):
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auc = roc_auc_score(df["y_true"], df["y_prob"])
        is_carcm = name == "CARCM"
        color = CARCM_COLOR if is_carcm else _BASELINE_COLORS[bi % len(_BASELINE_COLORS)]
        lw = CARCM_LW if is_carcm else BASELINE_LW
        if not is_carcm:
            bi += 1
        ax.plot(fpr, tpr, lw=lw, color=color,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — held-out test set", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"roc_curves.{ext}"))
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bi = 0
    for i, (name, df) in enumerate(preds.items()):
        prec, rec, _ = precision_recall_curve(df["y_true"], df["y_prob"])
        is_carcm = name == "CARCM"
        color = CARCM_COLOR if is_carcm else _BASELINE_COLORS[bi % len(_BASELINE_COLORS)]
        lw = CARCM_LW if is_carcm else BASELINE_LW
        if not is_carcm:
            bi += 1
        ax.plot(rec, prec, lw=lw, color=color, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"pr_curves.{ext}"))
    plt.close(fig)


def plot_hard_vs_easy(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.38
    bars_h = ax.bar(x - w / 2, df["hard_auc"], w, label="Hard test AUC",
                    color="#ff7f0e", edgecolor="black", lw=0.4)
    bars_e = ax.bar(x + w / 2, df["easy_auc"], w, label="Easy test AUC",
                    color="#2a9d8f", edgecolor="black", lw=0.4)
    for b, v in zip(bars_h, df["hard_auc"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars_e, df["easy_auc"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1.0)
    ax.set_title("Performance on hard (conflicting-fingerprint) vs easy test subsets",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"hard_vs_easy.{ext}"))
    plt.close(fig)


def plot_calibration(preds):
    fig, ax = plt.subplots(figsize=(7, 6))
    bi = 0
    for i, (name, df) in enumerate(preds.items()):
        prob_true, prob_pred = calibration_curve(df["y_true"], df["y_prob"],
                                                 n_bins=15, strategy="quantile")
        is_carcm = name == "CARCM"
        color = CARCM_COLOR if is_carcm else _BASELINE_COLORS[bi % len(_BASELINE_COLORS)]
        lw = CARCM_LW if is_carcm else BASELINE_LW
        if not is_carcm:
            bi += 1
        ax.plot(prob_pred, prob_true, marker="o", lw=lw, color=color, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=0.8)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"calibration.{ext}"))
    plt.close(fig)


def main():
    rows, preds = load_benchmark_results()
    CARCM_row, CARCM_pred = load_CARCM()
    rows.append(CARCM_row)
    if CARCM_pred is not None and {"y_true", "y_prob"}.issubset(CARCM_pred.columns):
        preds["CARCM"] = CARCM_pred.rename(
            columns={c: c for c in CARCM_pred.columns}
        )
    df = build_table(rows)
    print(df.to_string(index=False))
    write_table(df)
    plot_bars(df)
    plot_radar(df)
    plot_hard_vs_easy(df)
    if preds:
        plot_roc_pr(preds)
        plot_calibration(preds)
    print(f"\nAll artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
