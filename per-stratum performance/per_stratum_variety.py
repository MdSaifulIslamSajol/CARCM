#!/usr/bin/env python3
"""Table S-Variety: per-variety AUC with DeLong tests (CARCM vs Sequence-only)."""

import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from stratum_utils import (load_test_data, train_sequence_only_baseline,
                            compute_stratum_table, fmt_p, OUT_BASE, MIN_N_FOR_AUC)

OUT_DIR = os.path.join(OUT_BASE, "variety")
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "figure.dpi": 400,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def plot_variety_bar(df, out_dir):
    valid = df.dropna(subset=["carcm_auc"]).sort_values("carcm_auc", ascending=True)
    n = len(valid)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.55)))
    y = np.arange(n)
    w = 0.35

    bars_c = ax.barh(y + w / 2, valid["carcm_auc"], w, label="CARCM",
                     color="#2a9d8f", edgecolor="black", linewidth=0.5)
    bars_s = ax.barh(y - w / 2, valid["seq_only_auc"], w, label="Sequence-only",
                     color="#e9c46a", edgecolor="black", linewidth=0.5)

    for i, (_, row) in enumerate(valid.iterrows()):
        sig = ""
        if not pd.isna(row["delong_p"]) and row["delong_p"] < 0.05:
            sig = " *" if row["delong_p"] >= 0.01 else " **" if row["delong_p"] >= 0.001 else " ***"
        ax.text(row["carcm_auc"] + 0.005, i + w / 2,
                f'{row["carcm_auc"]:.3f}{sig}', va="center", fontsize=9)
        ax.text(row["seq_only_auc"] + 0.005, i - w / 2,
                f'{row["seq_only_auc"]:.3f}', va="center", fontsize=9, color="#666")

    ax.set_yticks(y)
    ax.set_yticklabels([f'{r["variety"]}  (n={r["n_test"]})' for _, r in valid.iterrows()],
                       fontsize=10)
    ax.set_xlabel("Test AUC", fontsize=13)
    ax.set_title("Per-Variety AUC: CARCM vs Sequence-Only Baseline", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, min(1.0, valid["carcm_auc"].max() + 0.12)])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"table_s_variety_bar.{ext}"))
    plt.close(fig)


def main():
    print("=" * 70)
    print("TABLE S-VARIETY: Per-Variety Performance")
    print("=" * 70)

    test_df, _, _, _ = load_test_data()
    print(f"\nTest samples: {len(test_df)}")
    print(f"Varieties: {test_df['variety'].nunique()}")

    print("\nTraining sequence-only baseline (LogReg, no context features)...")
    seq_probs = train_sequence_only_baseline()

    print("Computing per-variety metrics...")
    result = compute_stratum_table(test_df, "variety", seq_probs)

    result.to_csv(os.path.join(OUT_DIR, "table_s_variety.csv"),
                  index=False, float_format="%.4f")

    print(f"\n{'='*70}")
    print("TABLE S-VARIETY")
    print(f"{'='*70}")
    display = result.copy()
    display["delong_p"] = display["delong_p"].apply(fmt_p)
    display["carcm_auc"] = display["carcm_auc"].apply(
        lambda v: f"{v:.4f}" if not pd.isna(v) else f"n<{MIN_N_FOR_AUC}")
    display["seq_only_auc"] = display["seq_only_auc"].apply(
        lambda v: f"{v:.4f}" if not pd.isna(v) else f"n<{MIN_N_FOR_AUC}")
    display["delta_auc"] = display["delta_auc"].apply(
        lambda v: f"{v:+.4f}" if not pd.isna(v) else "\u2014")
    print(display.to_string(index=False))

    plot_variety_bar(result, OUT_DIR)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
