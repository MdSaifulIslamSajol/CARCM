#!/usr/bin/env python3
"""Statistical significance tests for the CARCM vs. benchmark comparison.

Produces, for every benchmark vs. CARCM (ours):
  - DeLong test on paired ROC AUCs (Z, p-value, AUC diff, 95% CI for diff)
  - McNemar test on paired binary decisions at each model's optimal threshold
  - Stratified bootstrap 95% CIs on AUC / PR-AUC / F1 / MCC (2000 resamples)
  - Benjamini–Hochberg FDR correction across all pairwise comparisons

Outputs (benchmarks/statistical_test_results/):
  - delong_vs_CARCM.csv
  - mcnemar_vs_CARCM.csv
  - bootstrap_ci.csv
  - pairwise_pvalues_bh.csv   (supplementary: raw p + BH q-values)
  - summary.csv               (merged view for the paper)
  - forest_auc_ci.{png,pdf}
  - pvalue_heatmap.{png,pdf}
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef)


RESULTS_DIR = "/home/saiful/CARCM/benchmarks/results"
CARCM_PRED = "/home/saiful/CARCM/models_carcm/best/best_predictions.csv"
CARCM_JSON = "/home/saiful/CARCM/models_carcm/best/best_results.json"
OUT_DIR = "/home/saiful/CARCM/benchmarks/statistical_test_results"



os.makedirs(OUT_DIR, exist_ok=True)

CARCM_NAME = "CARCM"
N_BOOT = 2000
RNG = np.random.default_rng(42)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# DeLong's test (Sun & Xu, 2014 fast implementation).
# Adapted implementation — tested against published reference values.
# ---------------------------------------------------------------------------

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
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based mean rank
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_by_label, label_1_count):
    m = label_1_count
    n = preds_sorted_by_label.shape[1] - m
    positive = preds_sorted_by_label[:, :m]
    negative = preds_sorted_by_label[:, m:]
    k = preds_sorted_by_label.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_by_label[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Return (auc_a, auc_b, z, p_value, ci_low, ci_high) for AUC(a)-AUC(b)."""
    order = (-y_true).argsort()
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())
    preds = np.vstack([prob_a[order], prob_b[order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    # cov is 2x2 when k=2
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var_diff <= 0:
        z = 0.0; p = 1.0; se = 0.0
    else:
        se = float(np.sqrt(var_diff))
        z = float(diff / se)
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    return float(aucs[0]), float(aucs[1]), z, p, float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# McNemar's test on paired binary decisions
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n01 = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    n10 = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    # Exact binomial p-value for small counts; continuity-corrected chi2 otherwise.
    if n01 + n10 < 25:
        p = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
        stat = float(min(n01, n10))
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p = float(1 - stats.chi2.cdf(stat, df=1))
    return n01, n10, float(stat), p


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_prob, threshold, n_boot=N_BOOT, alpha=0.05):
    """Stratified bootstrap: resample within each class to preserve prevalence."""
    y_pred = (y_prob >= threshold).astype(int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    aucs, praucs, f1s, mccs = [], [], [], []
    for _ in range(n_boot):
        bp = pos_idx[RNG.integers(0, n_pos, size=n_pos)]
        bn = neg_idx[RNG.integers(0, n_neg, size=n_neg)]
        idx = np.concatenate([bp, bn])
        yt = y_true[idx]; yp = y_prob[idx]; pr = y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        praucs.append(average_precision_score(yt, yp))
        f1s.append(f1_score(yt, pr, zero_division=0))
        mccs.append(matthews_corrcoef(yt, pr))

    def ci(a):
        a = np.asarray(a)
        lo = float(np.quantile(a, alpha / 2))
        hi = float(np.quantile(a, 1 - alpha / 2))
        return float(np.mean(a)), lo, hi

    auc_m, auc_lo, auc_hi = ci(aucs)
    prauc_m, prauc_lo, prauc_hi = ci(praucs)
    f1_m, f1_lo, f1_hi = ci(f1s)
    mcc_m, mcc_lo, mcc_hi = ci(mccs)
    return dict(auc_mean=auc_m, auc_lo=auc_lo, auc_hi=auc_hi,
                prauc_mean=prauc_m, prauc_lo=prauc_lo, prauc_hi=prauc_hi,
                f1_mean=f1_m, f1_lo=f1_lo, f1_hi=f1_hi,
                mcc_mean=mcc_m, mcc_lo=mcc_lo, mcc_hi=mcc_hi)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all():
    entries = {}
    for name in sorted(os.listdir(RESULTS_DIR)):
        mpath = os.path.join(RESULTS_DIR, name, "metrics.json")
        ppath = os.path.join(RESULTS_DIR, name, "predictions.csv")
        if not (os.path.exists(mpath) and os.path.exists(ppath)):
            continue
        m = json.load(open(mpath))
        p = pd.read_csv(ppath)
        entries[m["model"]] = dict(
            y_true=p["y_true"].values.astype(int),
            y_prob=p["y_prob"].values.astype(float),
            thr=float(m.get("optimal_threshold", 0.5)),
        )
    # CARCM
    vp = pd.read_csv(CARCM_PRED).rename(
        columns={"true_label": "y_true", "predicted_prob": "y_prob"}
    )
    vj = json.load(open(CARCM_JSON))
    entries[CARCM_NAME] = dict(
        y_true=vp["y_true"].values.astype(int),
        y_prob=vp["y_prob"].values.astype(float),
        thr=float(vj["metrics"]["optimal_threshold"]),
    )
    return entries


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_forest(df):
    df = df.sort_values("auc_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(df))
    colors = ["#e63946" if "ours" in m else "#264653" for m in df["model"]]
    ax.errorbar(
        df["auc_mean"], y,
        xerr=[df["auc_mean"] - df["auc_lo"], df["auc_hi"] - df["auc_mean"]],
        fmt="o", color="black", ecolor="gray", capsize=3, lw=1.2,
        markerfacecolor="white", markersize=7,
    )
    for i, c in enumerate(colors):
        ax.plot(df["auc_mean"].iloc[i], y[i], "o",
                color=c, markersize=9, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Test AUC (95% bootstrap CI, 2000 resamples)")
    ax.set_title("AUC point estimate and 95% CI per model", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"forest_auc_ci.{ext}"))
    plt.close(fig)


def plot_pvalue_heatmap(delong_df, mcnemar_df):
    models = list(delong_df["model"])
    mat = np.vstack([
        delong_df["delong_p"].values,
        mcnemar_df["mcnemar_p"].values,
    ])
    mat_log = -np.log10(np.clip(mat, 1e-300, 1.0))

    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(mat_log, aspect="auto", cmap="viridis")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["DeLong", "McNemar"])
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_title("Significance of CARCM vs. each benchmark  (−log10 p)",
                 fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            p = mat[i, j]
            txt = "<1e-300" if p < 1e-300 else f"{p:.1e}"
            ax.text(j, i, txt, ha="center", va="center",
                    color=("white" if mat_log[i, j] > 5 else "black"),
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="−log10(p)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"pvalue_heatmap.{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    entries = load_all()
    assert CARCM_NAME in entries, "CARCM predictions not found"
    CARCM = entries[CARCM_NAME]
    y_true = CARCM["y_true"]

    # Sanity: all predictions must share the same y_true ordering.
    for name, e in entries.items():
        if len(e["y_true"]) != len(y_true) or not np.array_equal(e["y_true"], y_true):
            raise RuntimeError(
                f"{name} has different y_true ordering than CARCM — cannot pair."
            )

    CARCM_pred = (CARCM["y_prob"] >= CARCM["thr"]).astype(int)

    delong_rows, mcnemar_rows, boot_rows = [], [], []
    for name, e in entries.items():
        ci = bootstrap_ci(e["y_true"], e["y_prob"], e["thr"])
        ci["model"] = name
        boot_rows.append(ci)

        if name == CARCM_NAME:
            continue

        auc_CARCM, auc_b, z, p_dl, lo, hi = delong_roc_test(
            y_true, CARCM["y_prob"], e["y_prob"]
        )
        delong_rows.append(dict(
            model=name,
            auc_CARCM=auc_CARCM, auc_benchmark=auc_b,
            auc_diff=auc_CARCM - auc_b,
            delong_z=z, delong_p=p_dl,
            ci_low=lo, ci_high=hi,
        ))

        b_pred = (e["y_prob"] >= e["thr"]).astype(int)
        n01, n10, stat, p_mc = mcnemar_test(y_true, CARCM_pred, b_pred)
        mcnemar_rows.append(dict(
            model=name,
            CARCM_right_bench_wrong=n01,
            CARCM_wrong_bench_right=n10,
            mcnemar_stat=stat, mcnemar_p=p_mc,
        ))

    order = ["GaussianNB", "BernoulliNB", "KNN", "DecisionTree",
             "LogisticRegression", "SVM_RBF", "RandomForest",
             "XGBoost", "LightGBM", "CatBoost", "TabNet", "FTTransformer"]

    def sort_df(df, col="model"):
        df["__o"] = df[col].apply(lambda m: order.index(m) if m in order else 99)
        return df.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    delong_df = sort_df(pd.DataFrame(delong_rows))
    mcnemar_df = sort_df(pd.DataFrame(mcnemar_rows))
    boot_order = order + [CARCM_NAME]
    boot_df = pd.DataFrame(boot_rows)
    boot_df["__o"] = boot_df["model"].apply(lambda m: boot_order.index(m) if m in boot_order else 99)
    boot_df = boot_df.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    boot_df = boot_df[["model", "auc_mean", "auc_lo", "auc_hi",
                       "prauc_mean", "prauc_lo", "prauc_hi",
                       "f1_mean", "f1_lo", "f1_hi",
                       "mcc_mean", "mcc_lo", "mcc_hi"]]

    # -------------------------------------------------------------------
    # Benjamini–Hochberg FDR correction across all baseline comparisons
    # -------------------------------------------------------------------
    all_raw_p = list(delong_df["delong_p"].values) + list(mcnemar_df["mcnemar_p"].values)
    all_raw_p = np.array([max(p, 1e-300) for p in all_raw_p])
    _, q_vals, _, _ = multipletests(all_raw_p, alpha=0.05, method="fdr_bh")

    n_baselines = len(delong_df)
    delong_df["delong_q_bh"] = q_vals[:n_baselines]
    mcnemar_df["mcnemar_q_bh"] = q_vals[n_baselines:]

    delong_df.to_csv(os.path.join(OUT_DIR, "delong_vs_CARCM.csv"),
                     index=False, float_format="%.6g")
    mcnemar_df.to_csv(os.path.join(OUT_DIR, "mcnemar_vs_CARCM.csv"),
                      index=False, float_format="%.6g")
    boot_df.to_csv(os.path.join(OUT_DIR, "bootstrap_ci.csv"),
                   index=False, float_format="%.4f")

    # -------------------------------------------------------------------
    # Supplementary table: pairwise p-values with BH-adjusted q-values
    # -------------------------------------------------------------------
    pairwise = delong_df[["model", "delong_p", "delong_q_bh"]].merge(
        mcnemar_df[["model", "mcnemar_p", "mcnemar_q_bh"]],
        on="model",
    )
    pairwise.to_csv(os.path.join(OUT_DIR, "pairwise_pvalues_bh.csv"),
                    index=False, float_format="%.6g")

    # -------------------------------------------------------------------
    # Combined summary (paper-ready)
    # -------------------------------------------------------------------
    summary = boot_df.copy()
    summary["auc_95ci"] = summary.apply(
        lambda r: f"{r.auc_mean:.4f} [{r.auc_lo:.4f}, {r.auc_hi:.4f}]", axis=1)
    summary["prauc_95ci"] = summary.apply(
        lambda r: f"{r.prauc_mean:.4f} [{r.prauc_lo:.4f}, {r.prauc_hi:.4f}]", axis=1)
    summary["f1_95ci"] = summary.apply(
        lambda r: f"{r.f1_mean:.4f} [{r.f1_lo:.4f}, {r.f1_hi:.4f}]", axis=1)
    summary["mcc_95ci"] = summary.apply(
        lambda r: f"{r.mcc_mean:.4f} [{r.mcc_lo:.4f}, {r.mcc_hi:.4f}]", axis=1)
    merged = summary[["model", "auc_95ci", "prauc_95ci", "f1_95ci", "mcc_95ci"]].merge(
        delong_df[["model", "auc_diff", "delong_z", "delong_p", "delong_q_bh"]],
        on="model", how="left",
    ).merge(
        mcnemar_df[["model", "mcnemar_stat", "mcnemar_p", "mcnemar_q_bh"]],
        on="model", how="left",
    )
    merged.to_csv(os.path.join(OUT_DIR, "summary.csv"),
                  index=False, float_format="%.6g")

    plot_forest(boot_df)
    plot_pvalue_heatmap(delong_df, mcnemar_df)

    print("\n=== DeLong (CARCM vs benchmark) — with BH q-values ===")
    print(delong_df.to_string(index=False))
    print("\n=== McNemar (CARCM vs benchmark) — with BH q-values ===")
    print(mcnemar_df.to_string(index=False))
    print("\n=== Stratified Bootstrap CIs (AUC, PR-AUC, F1, MCC) ===")
    print(boot_df.to_string(index=False))
    print("\n=== Supplementary: Pairwise p-values with BH-adjusted q ===")
    print(pairwise.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
