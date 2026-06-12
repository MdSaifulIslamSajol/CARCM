#!/usr/bin/env python3
"""Dataset overview figure — 16:9 layout, 2x2 grid, 400 dpi."""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

INPUT = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/data_preprocess/RiceMetaSys_Drought_merged_all_excels_6.csv"
OUT = "/home/saiful/CARCM/benchmarks/dataset_overview4.png"

VARIETY_RENAME = {
    "IR64_Cleaned": "IR64",
    "N22_Cleaned3": "N22",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def set3(n):
    return plt.cm.Set3(np.linspace(0, 1, max(n, 3)))


def main():
    df = pd.read_csv(INPUT).dropna(subset=["promoter_sequence"]).reset_index(drop=True)
    df["Variety"] = df["Variety"].replace(VARIETY_RENAME)

    # 16:9 figsize — 16 x 9 inches gives exact 16:9 at any DPI
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axA, axB, axC, axD = axes.flat

    # ---------- A. Class distribution ----------
    cls = df["Gene Regulation"].value_counts()
    class_labels = ["Up-regulated", "Down-regulated"]
    class_vals = [cls.get("Up Regulated", 0), cls.get("Down Regulated", 0)]
    colors_cls = set3(4)[:2]
    bars = axA.bar(class_labels, class_vals, color=colors_cls,
                   edgecolor="black", linewidth=0.6)
    for b in bars:
        axA.text(b.get_x() + b.get_width() / 2, b.get_height() + 250,
                 f"{int(b.get_height()):,}",
                 ha="center", fontsize=10, fontweight="bold")
    axA.set_ylabel("# samples")
    axA.set_title("A. Class distribution")
    axA.set_ylim(0, max(class_vals) * 1.15)

    # ---------- B. Varieties (horizontal) ----------
    var_counts = df["Variety"].value_counts()
    colors_var = set3(len(var_counts))
    y_pos = np.arange(len(var_counts))
    axB.barh(y_pos, var_counts.values,
             color=colors_var, edgecolor="black", linewidth=0.4)
    axB.set_yticks(y_pos)
    axB.set_yticklabels(var_counts.index)
    axB.invert_yaxis()
    axB.set_xlabel("# samples")
    axB.set_title(f"B. Varieties (n = {len(var_counts)})")
    axB.set_xlim(0, var_counts.values.max() * 1.18)
    for yi, v in zip(y_pos, var_counts.values):
        axB.text(v + var_counts.values.max() * 0.01, yi, f"{v:,}",
                 va="center", fontsize=8)

    # ---------- C. Tissues ----------
    tissue_counts = df["Tissue"].value_counts()
    colors_tis = set3(len(tissue_counts))
    axC.bar(range(len(tissue_counts)), tissue_counts.values,
            color=colors_tis, edgecolor="black", linewidth=0.4)
    axC.set_xticks(range(len(tissue_counts)))
    axC.set_xticklabels(tissue_counts.index, rotation=35, ha="right")
    axC.set_ylabel("# samples")
    axC.set_title(f"C. Tissues (n = {len(tissue_counts)})")
    for i, v in enumerate(tissue_counts.values):
        axC.text(i, v + 150, f"{v:,}", ha="center", fontsize=8)
    axC.set_ylim(0, max(tissue_counts.values) * 1.15)

    # ---------- D. Developmental stages ----------
    stage_counts = df["Stage"].value_counts()
    colors_sta = set3(len(stage_counts))
    axD.bar(range(len(stage_counts)), stage_counts.values,
            color=colors_sta, edgecolor="black", linewidth=0.4)
    axD.set_xticks(range(len(stage_counts)))
    axD.set_xticklabels(
        [s.replace(" Stage", "") for s in stage_counts.index],
        rotation=35, ha="right",
    )
    axD.set_ylabel("# samples")
    axD.set_title(f"D. Developmental stages (n = {len(stage_counts)})")
    for i, v in enumerate(stage_counts.values):
        axD.text(i, v + 300, f"{v:,}", ha="center", fontsize=8)
    axD.set_ylim(0, max(stage_counts.values) * 1.15)

    fig.suptitle(
        f"Rice Drought Promoter Dataset — Overview "
        f"(N = {len(df):,} promoter × context samples)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=400)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
