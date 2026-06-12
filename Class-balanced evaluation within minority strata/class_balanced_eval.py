#!/usr/bin/env python3
"""
Class-Balanced Evaluation Within Minority Strata
=================================================
Checks whether CARCM is genuinely discriminating within each stratum,
or merely exploiting the stratum's class prior.

For every stratum (variety, tissue, stage) with both classes present:
  - AUC  (ranking quality — immune to class imbalance)
  - Balanced Accuracy  (mean of per-class recall — 0.5 = random)
  - MCC  (–1 to +1, 0 = random — handles imbalance correctly)

A stratum where balanced-acc ≈ 0.5 but AUC ≈ 0.7 is a ranking artefact:
the model ranks positives above negatives on average but at the chosen
threshold it predicts the majority label for all/most samples.
"""

import os, sys, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             matthews_corrcoef, confusion_matrix,
                             accuracy_score)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/saiful/CARCM/output"
CARCM_PRED = "/home/saiful/CARCM/models_carcm/best/best_predictions.csv"
OUT_BASE = "/home/saiful/CARCM/Class-balanced evaluation within minority strata"
THRESHOLD = 0.3

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 400,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_data():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    pred_df = pd.read_csv(CARCM_PRED)

    var_cols = sorted([c for c in test_df.columns if c.startswith("Variety_")])
    tis_cols = sorted([c for c in test_df.columns if c.startswith("Tissue_")])
    stg_cols = sorted([c for c in test_df.columns if c.startswith("Stage_")])

    test_df["variety"] = test_df[var_cols].idxmax(axis=1).str.replace("Variety_", "", regex=False)
    test_df["tissue"] = test_df[tis_cols].idxmax(axis=1).str.replace("Tissue_", "", regex=False)
    test_df["stage"] = test_df[stg_cols].idxmax(axis=1).str.replace("Stage_", "", regex=False)
    test_df["true_label"] = pred_df["true_label"].astype(int).values
    test_df["carcm_prob"] = pred_df["predicted_prob"].values
    test_df["carcm_pred"] = (pred_df["predicted_prob"] >= THRESHOLD).astype(int).values

    return test_df


def evaluate_stratum(y_true, y_prob, y_pred):
    """Return dict of class-balanced metrics for one stratum."""
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    minority_n = min(n_pos, n_neg)
    both_classes = len(np.unique(y_true)) == 2

    result = {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "minority_n": minority_n,
        "class_ratio": round(n_pos / max(n_neg, 1), 2),
        "imbalance_ratio": round(max(n_pos, n_neg) / max(minority_n, 1), 2),
    }

    if both_classes:
        result["auc"] = round(roc_auc_score(y_true, y_prob), 4)
        result["balanced_acc"] = round(balanced_accuracy_score(y_true, y_pred), 4)
        result["mcc"] = round(matthews_corrcoef(y_true, y_pred), 4)
        result["accuracy"] = round(accuracy_score(y_true, y_pred), 4)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        result["sensitivity"] = round(tp / max(tp + fn, 1), 4)
        result["specificity"] = round(tn / max(tn + fp, 1), 4)
        result["tp"] = int(tp)
        result["fp"] = int(fp)
        result["fn"] = int(fn)
        result["tn"] = int(tn)
    else:
        for k in ["auc", "balanced_acc", "mcc", "accuracy",
                   "sensitivity", "specificity", "tp", "fp", "fn", "tn"]:
            result[k] = np.nan

    # Diagnostic flag
    if both_classes and result["auc"] >= 0.6 and result["balanced_acc"] < 0.55:
        result["artefact_flag"] = "POSSIBLE ARTEFACT"
    elif both_classes and result["mcc"] < 0.1 and result["auc"] >= 0.6:
        result["artefact_flag"] = "POSSIBLE ARTEFACT"
    else:
        result["artefact_flag"] = ""

    return result


def analyze_all_strata(df):
    """Run class-balanced eval across variety, tissue, stage."""
    all_results = []

    for stratum_type, col in [("Variety", "variety"),
                               ("Tissue", "tissue"),
                               ("Stage", "stage")]:
        for name, grp in df.groupby(col):
            y = grp["true_label"].values
            p = grp["carcm_prob"].values
            pred = grp["carcm_pred"].values

            r = evaluate_stratum(y, p, pred)
            r["stratum_type"] = stratum_type
            r["stratum"] = name
            all_results.append(r)

    result_df = pd.DataFrame(all_results)
    cols = ["stratum_type", "stratum", "n", "n_pos", "n_neg", "minority_n",
            "class_ratio", "imbalance_ratio", "auc", "balanced_acc", "mcc",
            "accuracy", "sensitivity", "specificity",
            "tp", "fp", "fn", "tn", "artefact_flag"]
    return result_df[cols].sort_values(["stratum_type", "n"],
                                        ascending=[True, False]).reset_index(drop=True)


def plot_balanced_vs_auc(df, out_dir):
    """Scatter: AUC vs Balanced Accuracy, sized by n, coloured by stratum type."""
    valid = df.dropna(subset=["auc", "balanced_acc"]).copy()

    colors = {"Variety": "#2a9d8f", "Tissue": "#264653", "Stage": "#e76f51"}
    markers = {"Variety": "o", "Tissue": "s", "Stage": "D"}

    fig, ax = plt.subplots(figsize=(10, 7))

    for stype in ["Variety", "Tissue", "Stage"]:
        sub = valid[valid["stratum_type"] == stype]
        sizes = np.clip(sub["n"] / 10, 30, 300)
        ax.scatter(sub["auc"], sub["balanced_acc"], s=sizes,
                   c=colors[stype], marker=markers[stype],
                   alpha=0.75, edgecolors="black", linewidth=0.5,
                   label=stype, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["stratum"], (row["auc"], row["balanced_acc"]),
                        fontsize=6.5, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label="Balanced Acc = 0.5 (random)")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.fill_between([0.6, 1.0], 0, 0.55, color="red", alpha=0.06, zorder=0)
    ax.text(0.82, 0.52, "Artefact zone\n(high AUC, low Bal-Acc)",
            fontsize=8, color="red", alpha=0.7, ha="center", style="italic")

    ax.set_xlabel("AUC (Ranking Quality)", fontsize=13)
    ax.set_ylabel("Balanced Accuracy (Classification Quality)", fontsize=13)
    ax.set_title("Class-Balanced Evaluation: AUC vs Balanced Accuracy\n"
                 "(bubble size \u221d sample size)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0.4, 1.02])
    ax.set_ylim([0.35, 1.02])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"auc_vs_balanced_acc.{ext}"))
    plt.close(fig)


def plot_mcc_bar(df, out_dir):
    """Horizontal bar chart of MCC across all strata."""
    valid = df.dropna(subset=["mcc"]).sort_values("mcc", ascending=True).copy()
    valid["label"] = valid["stratum_type"] + ": " + valid["stratum"]

    n = len(valid)
    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.35)))

    colors_map = {"Variety": "#2a9d8f", "Tissue": "#264653", "Stage": "#e76f51"}
    bar_colors = [colors_map[t] for t in valid["stratum_type"]]

    bars = ax.barh(range(n), valid["mcc"], color=bar_colors,
                   edgecolor="black", linewidth=0.4)

    for i, (_, row) in enumerate(valid.iterrows()):
        offset = 0.02 if row["mcc"] >= 0 else -0.02
        ha = "left" if row["mcc"] >= 0 else "right"
        ax.text(row["mcc"] + offset, i, f'{row["mcc"]:.3f}  (n={row["n"]})',
                va="center", fontsize=7.5, ha=ha)

    ax.set_yticks(range(n))
    ax.set_yticklabels(valid["label"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
    ax.set_title("Per-Stratum MCC: Class-Balanced Discrimination Quality",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for l, c in colors_map.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"mcc_per_stratum.{ext}"))
    plt.close(fig)


def plot_sensitivity_specificity(df, out_dir):
    """Paired dot plot: sensitivity vs specificity per stratum."""
    valid = df.dropna(subset=["sensitivity"]).sort_values(
        ["stratum_type", "balanced_acc"], ascending=[True, False]).copy()
    valid["label"] = valid["stratum_type"].str[0] + ": " + valid["stratum"]

    n = len(valid)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
    y = np.arange(n)

    ax.scatter(valid["sensitivity"], y, color="#e63946", s=60,
               zorder=3, label="Sensitivity (TPR)", marker="o", edgecolors="black", linewidth=0.4)
    ax.scatter(valid["specificity"], y, color="#457b9d", s=60,
               zorder=3, label="Specificity (TNR)", marker="D", edgecolors="black", linewidth=0.4)

    for i in range(n):
        sens = valid.iloc[i]["sensitivity"]
        spec = valid.iloc[i]["specificity"]
        ax.plot([sens, spec], [i, i], color="gray", linewidth=0.6, alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(valid["label"], fontsize=7.5)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Rate", fontsize=12)
    ax.set_title("Per-Stratum Sensitivity vs Specificity\n(at threshold = 0.3)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0, 1.05])
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"sensitivity_specificity.{ext}"))
    plt.close(fig)


def print_conclusion(df):
    """Analyse results and print honest conclusion."""
    valid = df.dropna(subset=["auc"]).copy()

    artefacts = valid[valid["artefact_flag"] != ""]
    low_mcc = valid[valid["mcc"] < 0.1]
    low_balacc = valid[valid["balanced_acc"] < 0.55]

    minority_strata = valid[valid["minority_n"] < 50].copy()
    well_powered = valid[valid["minority_n"] >= 50].copy()

    print(f"\n{'='*75}")
    print("CONCLUSION: CLASS-BALANCED EVALUATION")
    print(f"{'='*75}")

    print(f"\n  Total strata analysed: {len(valid)} (with both classes present)")
    print(f"  Well-powered strata (minority class n >= 50): {len(well_powered)}")
    print(f"  Low-power strata (minority class n < 50): {len(minority_strata)}")

    print(f"\n  --- Well-Powered Strata (minority n >= 50) ---")
    if len(well_powered) > 0:
        print(f"  AUC range:          [{well_powered['auc'].min():.4f}, {well_powered['auc'].max():.4f}]")
        print(f"  Balanced Acc range: [{well_powered['balanced_acc'].min():.4f}, {well_powered['balanced_acc'].max():.4f}]")
        print(f"  MCC range:          [{well_powered['mcc'].min():.4f}, {well_powered['mcc'].max():.4f}]")
        print(f"  All MCC > 0.1:      {(well_powered['mcc'] > 0.1).all()}")
        print(f"  All Bal-Acc > 0.55: {(well_powered['balanced_acc'] > 0.55).all()}")

    print(f"\n  --- Low-Power / Minority Strata (minority n < 50) ---")
    if len(minority_strata) > 0:
        for _, row in minority_strata.iterrows():
            flag = f"  *** {row['artefact_flag']} ***" if row["artefact_flag"] else ""
            print(f"  {row['stratum_type']:8s}: {row['stratum']:25s}  "
                  f"n={row['n']:4d}  minority={row['minority_n']:3d}  "
                  f"AUC={row['auc']:.3f}  BalAcc={row['balanced_acc']:.3f}  "
                  f"MCC={row['mcc']:.3f}  Sens={row['sensitivity']:.3f}  "
                  f"Spec={row['specificity']:.3f}{flag}")
    else:
        print("  (none)")

    print(f"\n  --- Artefact Detection ---")
    if len(artefacts) > 0:
        print(f"  WARNING: {len(artefacts)} stratum/strata flagged as possible artefact:")
        for _, row in artefacts.iterrows():
            print(f"    {row['stratum_type']}: {row['stratum']}  "
                  f"(AUC={row['auc']:.3f}, BalAcc={row['balanced_acc']:.3f}, "
                  f"MCC={row['mcc']:.3f})")
        print("  These strata should be interpreted with caution.")
    else:
        print("  No artefacts detected. In all strata where AUC > 0.6,")
        print("  balanced accuracy and MCC confirm genuine discrimination.")

    print(f"\n  --- Key Takeaway ---")
    if len(artefacts) == 0 and (well_powered["mcc"] > 0.1).all():
        print("  CARCM demonstrates genuine class discrimination across all well-powered")
        print("  strata. Balanced accuracy consistently exceeds 0.5 (random baseline),")
        print("  and MCC values are positive, confirming the model is not merely")
        print("  predicting the majority class label within each stratum.")
        if len(minority_strata) > 0:
            print(f"\n  For the {len(minority_strata)} low-power strata (minority n < 50),")
            print("  results should be interpreted cautiously due to limited sample size.")
    else:
        print("  Some strata show signs of ranking artefact. CARCM's AUC may be")
        print("  inflated in those strata — the model ranks samples reasonably but")
        print("  at the decision threshold it collapses to majority-class prediction.")


def main():
    print("=" * 75)
    print("CLASS-BALANCED EVALUATION WITHIN MINORITY STRATA")
    print("=" * 75)
    print(f"\nDecision threshold: {THRESHOLD}")

    df = load_data()
    print(f"Test samples: {len(df)}")

    results = analyze_all_strata(df)

    results.to_csv(os.path.join(OUT_BASE, "class_balanced_eval.csv"),
                   index=False, float_format="%.4f")
    print(f"\nSaved: class_balanced_eval.csv")

    # Print full table
    print(f"\n{'='*75}")
    print("FULL RESULTS TABLE")
    print(f"{'='*75}")
    display = results.copy()
    display_cols = ["stratum_type", "stratum", "n", "minority_n",
                    "imbalance_ratio", "auc", "balanced_acc", "mcc",
                    "sensitivity", "specificity", "artefact_flag"]
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 200)
    print(display[display_cols].to_string(index=False))

    # Plots
    print(f"\n{'='*75}")
    print("GENERATING FIGURES")
    print(f"{'='*75}")

    plot_balanced_vs_auc(results, OUT_BASE)
    print("  Saved: auc_vs_balanced_acc.{png,pdf}")

    plot_mcc_bar(results, OUT_BASE)
    print("  Saved: mcc_per_stratum.{png,pdf}")

    plot_sensitivity_specificity(results, OUT_BASE)
    print("  Saved: sensitivity_specificity.{png,pdf}")

    print_conclusion(results)

    print(f"\nAll outputs saved to:\n  {OUT_BASE}/")


if __name__ == "__main__":
    main()
