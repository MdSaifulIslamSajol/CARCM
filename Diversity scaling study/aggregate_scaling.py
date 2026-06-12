#!/usr/bin/env python3
"""
Aggregate the per-run CSVs produced by run_scaling.py into:
    - summary_<panel>.csv    (mean, std, 95% CI across seeds for A and B)
    - curves_<panel>.png     (400 dpi, two curves + shaded CI)
    - delta_curve_<panel>.png (Curve A - Curve B vs. sample count)
    - scaling_overview.png   (3-panel side-by-side figure)
    - delta_significance.csv (paired one-sided Wilcoxon signed-rank)
    - full_model_reference.json (AUC of the full CARCM model if available)

Run after all panels finish:
    conda run -n dnabert python aggregate_scaling.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANELS = ["variety", "tissue", "stage"]
PANEL_TITLES = {
    "variety": "Variety scaling",
    "tissue":  "Tissue scaling",
    "stage":   "Stage scaling",
}

# Matplotlib publication defaults
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

CURVE_COLORS = {"A": "#1f77b4", "B": "#d62728"}
CURVE_LABELS = {"A": "Diversity (k contexts)", "B": "Sample-matched control (1 context)"}


# =============================================================================

def load_full_model_auc() -> float | None:
    """Try to pull the full-model test AUC from existing CARCM results."""
    candidates = [
        Path("/home/saiful/CARCM/models_carcm/best/best_results.json"),
        Path("/home/saiful/CARCM/benchmarks/results/summary.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                with open(path) as f:
                    js = json.load(f)
                for key in ("metrics", "test_metrics"):
                    if key in js and "test_auc" in js[key]:
                        return float(js[key]["test_auc"])
                if "test_auc" in js:
                    return float(js["test_auc"])
            elif path.name == "summary.csv":
                df = pd.read_csv(path)
                if "AUC" in df.columns and "Model" in df.columns:
                    row = df[df["Model"].str.contains("CARCM", case=False, na=False)]
                    if len(row):
                        val = row.iloc[0]["AUC"]
                        # Strip "0.XXX [lo-hi]" formatting if present.
                        if isinstance(val, str):
                            val = val.split()[0]
                        return float(val)
        except Exception:
            continue
    return None


def summarise_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean, std, and 95% CI (across seeds) per (curve, k)."""
    rows = []
    for (curve, k), g in df.groupby(["curve", "k"]):
        aucs = g["test_auc"].values
        n = len(aucs)
        mean = float(np.mean(aucs))
        std = float(np.std(aucs, ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0
        # 95 % CI via t-distribution (small n)
        if n > 1:
            tcrit = stats.t.ppf(0.975, df=n - 1)
            ci_lo = mean - tcrit * sem
            ci_hi = mean + tcrit * sem
        else:
            ci_lo = ci_hi = mean
        rows.append({
            "curve": curve,
            "k": k,
            "n_seeds": n,
            "auc_mean": mean,
            "auc_std": std,
            "auc_ci_lo": ci_lo,
            "auc_ci_hi": ci_hi,
            "n_train_mean": float(g["n_train"].mean()),
            "mcc_mean": float(g["test_mcc"].mean()),
            "f1_mean": float(g["test_f1"].mean()),
            "bal_acc_mean": float(g["test_bal_acc"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["curve", "k"]).reset_index(drop=True)


def paired_delta_test(df: pd.DataFrame) -> pd.DataFrame:
    """At each k, paired one-sided test that Curve A > Curve B on AUC.

    With only 3 seeds, the Wilcoxon signed-rank test has a discrete p-value
    floor of 1/(2**n) = 0.125 when every difference is positive, so it can
    never reject at alpha=0.05. We therefore also report a paired t-test
    (continuous p-value) and treat "delta CI excludes zero" as the primary
    significance criterion.
    """
    rows = []
    for k in sorted(df["k"].unique()):
        a = df[(df["curve"] == "A") & (df["k"] == k)].sort_values("seed")["test_auc"].values
        b = df[(df["curve"] == "B") & (df["k"] == k)].sort_values("seed")["test_auc"].values
        n = min(len(a), len(b))
        if n < 2:
            rows.append({"k": k, "n": n, "delta_mean": np.nan})
            continue
        a, b = a[:n], b[:n]
        deltas = a - b
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1))
        sem_d = std_d / math.sqrt(n) if n > 1 else 0.0
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci_lo = mean_d - tcrit * sem_d
        ci_hi = mean_d + tcrit * sem_d

        # Paired one-sided t-test (A > B)
        try:
            t_stat, p_two = stats.ttest_rel(a, b)
            if np.isnan(p_two):
                p_t = 1.0
            else:
                p_t = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        except Exception:
            p_t = 1.0

        # Wilcoxon (one-sided) for reference only
        try:
            _, p_w = stats.wilcoxon(a, b, alternative="greater")
            p_w = float(p_w)
        except Exception:
            p_w = 1.0

        # Primary significance: does the 95% CI of the paired delta exclude zero?
        ci_excludes_zero = bool(ci_lo > 0)

        rows.append({
            "k": k,
            "n": n,
            "delta_mean": mean_d,
            "delta_ci_lo": ci_lo,
            "delta_ci_hi": ci_hi,
            "p_paired_t_one_sided": float(p_t),
            "p_wilcoxon_one_sided": p_w,
            "ci_excludes_zero": ci_excludes_zero,
        })
    return pd.DataFrame(rows)


def plot_panel(ax, summary: pd.DataFrame, title: str, ref_auc: float | None):
    for curve in ["A", "B"]:
        sub = summary[summary["curve"] == curve].sort_values("k")
        ax.plot(
            sub["k"], sub["auc_mean"],
            marker="o", color=CURVE_COLORS[curve], lw=2.0,
            label=CURVE_LABELS[curve],
        )
        ax.fill_between(
            sub["k"], sub["auc_ci_lo"], sub["auc_ci_hi"],
            color=CURVE_COLORS[curve], alpha=0.2,
        )
    if ref_auc is not None:
        ax.axhline(ref_auc, color="black", linestyle="--", lw=1.5,
                   label=f"Full CARCM (AUC={ref_auc:.3f})")
    ax.set_title(title)
    ax.set_xlabel("k (number of contexts)")
    ax.set_ylabel("Test AUC")
    ax.grid(True, alpha=0.3)


def plot_delta(ax, summary: pd.DataFrame, title: str):
    a = summary[summary["curve"] == "A"].sort_values("k")
    b = summary[summary["curve"] == "B"].sort_values("k")
    merged = a.merge(b, on="k", suffixes=("_A", "_B"))
    merged["delta"] = merged["auc_mean_A"] - merged["auc_mean_B"]
    ax.bar(merged["k"], merged["delta"], color="#2ca02c", alpha=0.8,
           edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("Curve A - Curve B (AUC)")
    ax.grid(True, alpha=0.3)


# =============================================================================

def main():
    ref_auc = load_full_model_auc()
    print(f"Full-model reference AUC: {ref_auc}")

    panel_summaries = {}
    panel_deltas = {}

    for panel in PANELS:
        path = RESULTS_DIR / f"scaling_{panel}.csv"
        if not path.exists():
            print(f"!! Missing {path} — skipping.")
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            print(f"!! {path} is empty — skipping.")
            continue

        summary = summarise_panel(df)
        summary.to_csv(RESULTS_DIR / f"summary_{panel}.csv", index=False)
        panel_summaries[panel] = summary

        delta = paired_delta_test(df)
        delta.to_csv(RESULTS_DIR / f"delta_{panel}.csv", index=False)
        panel_deltas[panel] = delta

        # Per-panel single figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        plot_panel(ax, summary, PANEL_TITLES[panel], ref_auc)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"curve_{panel}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)

        # Per-panel delta figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
        plot_delta(ax, summary, f"Diversity benefit ({panel})")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"delta_{panel}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)

        print(f"[{panel}] summary rows: {len(summary)}")

    # Overview figure (three panels side by side) ==========================
    if panel_summaries:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        for ax, panel in zip(axes, PANELS):
            if panel not in panel_summaries:
                continue
            plot_panel(ax, panel_summaries[panel], PANEL_TITLES[panel], ref_auc)
        axes[-1].legend(loc="lower right", frameon=False)
        axes[0].set_ylabel("Test AUC")
        for ax in axes[1:]:
            ax.set_ylabel("")
        fig.suptitle("Training diversity scaling (CARCM)", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "scaling_overview.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved overview figure -> {FIGURES_DIR/'scaling_overview.png'}")

    # Save reference AUC for report use
    with open(RESULTS_DIR / "full_model_reference.json", "w") as f:
        json.dump({"full_model_auc": ref_auc}, f, indent=2)

    # Save concatenated delta significance table
    if panel_deltas:
        all_deltas = []
        for panel, d in panel_deltas.items():
            d = d.copy()
            d.insert(0, "panel", panel)
            all_deltas.append(d)
        concat = pd.concat(all_deltas, ignore_index=True)
        concat.to_csv(RESULTS_DIR / "delta_significance.csv", index=False)
        print(concat.to_string(index=False))


if __name__ == "__main__":
    main()
