#!/usr/bin/env python3
"""
Step 5: Context Ablation Ladder on Homology-Aware Split
========================================================
Train three variants on the homology-aware split:
  1. CARCM (full)    – already trained in step4
  2. Sequence-only   – zero out all context features
  3. Context-only    – zero out all sequence features
Computes AUC, PR-AUC, F1, MCC for each, then DeLong test CARCM vs seq-only.
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "5")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/saiful/CARCM")
from rice_classifier_v7b import (
    ContextEnhancedGatedFusionV5, PositionalMotifDataset,
    create_context_interactions, identify_hard_examples, load_feature_groups,
    NUM_BINS, set_seed, get_predictions
)

BASE = "/home/saiful/CARCM/Homology aware gene level split"
OUTPUT_DIR = os.path.join(BASE, "output")
MODELS_DIR = os.path.join(BASE, "checkpoints")
RESULTS_DIR = os.path.join(BASE, "ablation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def delong_pvalue(y_true, y_score1, y_score2):
    """Fast DeLong test for comparing two AUCs."""
    from scipy.stats import norm
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    pos = y_true == 1
    neg = y_true == 0

    V10_1 = np.array([np.mean(y_score1[pos] > t) + 0.5 * np.mean(y_score1[pos] == t)
                       for t in y_score1[neg]])
    V10_2 = np.array([np.mean(y_score2[pos] > t) + 0.5 * np.mean(y_score2[pos] == t)
                       for t in y_score2[neg]])
    V01_1 = np.array([np.mean(y_score1[neg] < t) + 0.5 * np.mean(y_score1[neg] == t)
                       for t in y_score1[pos]])
    V01_2 = np.array([np.mean(y_score2[neg] < t) + 0.5 * np.mean(y_score2[neg] == t)
                       for t in y_score2[pos]])

    auc1 = np.mean(V10_1)
    auc2 = np.mean(V10_2)

    S10 = np.cov(np.vstack([V10_1, V10_2]))
    S01 = np.cov(np.vstack([V01_1, V01_2]))

    S = S10 / n0 + S01 / n1
    z = (auc1 - auc2) / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1] + 1e-10)
    p = 2 * (1 - norm.cdf(abs(z)))
    return p


def train_logistic_baseline(train_df, test_df, feature_cols, label_col="label"):
    """Train a LogisticRegression baseline on given features."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    y_train = train_df[label_col].values
    y_test = test_df[label_col].values

    lr = LogisticRegression(max_iter=5000, C=1.0, random_state=42, solver="lbfgs")
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    return y_test, probs


def main():
    print("=" * 70)
    print("STEP 5: CONTEXT ABLATION LADDER — HOMOLOGY-AWARE SPLIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    feature_groups = load_feature_groups(OUTPUT_DIR)
    variety_cols = feature_groups["variety"]
    tissue_cols = feature_groups["tissue"]
    stage_cols = feature_groups["stage"]

    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
    dev_df = pd.read_csv(os.path.join(OUTPUT_DIR, "dev.csv"))
    test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "test.csv"))
    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    count_cols = feature_groups["global_count"]
    train_hard, test_hard, _ = identify_hard_examples(train_df, test_df, count_cols)

    context_info_train = create_context_interactions(train_df, variety_cols, tissue_cols, stage_cols)
    context_info_test = create_context_interactions(test_df, variety_cols, tissue_cols, stage_cols)

    train_dataset = PositionalMotifDataset(
        train_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info_train, train_hard, fit_scaler=True
    )
    test_dataset = PositionalMotifDataset(
        test_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info_test, test_hard, scaler=train_dataset.scaler
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # ── Load CARCM ensemble models ───────────────────────────────────────────
    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith("best_seed") and f.endswith(".pt")])
    if not model_files:
        print("ERROR: No trained CARCM models found. Run step4 first.")
        sys.exit(1)

    print(f"\nFound {len(model_files)} ensemble models")

    # Get model config from first checkpoint
    first_ckpt = torch.load(os.path.join(MODELS_DIR, model_files[0]), map_location=device)
    config = first_ckpt.get("config", {})
    hidden_dim = config.get("hidden_dim", 128)

    n_varieties = len(variety_cols)
    n_tissues = len(tissue_cols)
    n_stages = len(stage_cols)

    models = []
    for mf in model_files:
        ckpt = torch.load(os.path.join(MODELS_DIR, mf), map_location=device)
        model = ContextEnhancedGatedFusionV5(
            n_motifs=len(count_cols),
            n_bins=NUM_BINS,
            n_cooc=train_dataset.n_cooc,
            n_varieties=n_varieties,
            n_tissues=n_tissues,
            n_stages=n_stages,
            n_vt=context_info_train["n_vt"],
            n_ts=context_info_train["n_ts"],
            n_vs=context_info_train["n_vs"],
            hidden_dim=hidden_dim,
            dropout=config.get("dropout", 0.4),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    # ── Evaluate CARCM (full) ────────────────────────────────────────────────
    print("\n--- Evaluating CARCM (full model) ---")
    all_labels = test_df["label"].values
    all_preds_full = []
    for m in models:
        preds = get_predictions(m, test_loader, device)
        all_preds_full.append(preds)
    ensemble_preds_full = np.mean(all_preds_full, axis=0)

    auc_full = roc_auc_score(all_labels, ensemble_preds_full)
    pr_auc_full = average_precision_score(all_labels, ensemble_preds_full)
    thresh = 0.5
    f1_full = f1_score(all_labels, (ensemble_preds_full >= thresh).astype(int))
    mcc_full = matthews_corrcoef(all_labels, (ensemble_preds_full >= thresh).astype(int))
    ba_full = balanced_accuracy_score(all_labels, (ensemble_preds_full >= thresh).astype(int))
    print(f"  AUC={auc_full:.4f}, PR-AUC={pr_auc_full:.4f}, F1={f1_full:.4f}, "
          f"MCC={mcc_full:.4f}, BA={ba_full:.4f}")

    # ── Evaluate Sequence-only (zero context at inference) ───────────────────
    print("\n--- Evaluating Sequence-only (context zeroed at inference) ---")
    all_preds_seqonly = []
    for m in models:
        preds_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                batch_dev["variety"] = torch.zeros_like(batch_dev["variety"])
                batch_dev["tissue"] = torch.zeros_like(batch_dev["tissue"])
                batch_dev["stage"] = torch.zeros_like(batch_dev["stage"])
                if "variety_tissue" in batch_dev:
                    batch_dev["variety_tissue"] = torch.zeros_like(batch_dev["variety_tissue"])
                    batch_dev["tissue_stage"] = torch.zeros_like(batch_dev["tissue_stage"])
                    batch_dev["variety_stage"] = torch.zeros_like(batch_dev["variety_stage"])
                    batch_dev["variety_idx"] = torch.zeros_like(batch_dev["variety_idx"])
                    batch_dev["tissue_idx"] = torch.zeros_like(batch_dev["tissue_idx"])
                    batch_dev["stage_idx"] = torch.zeros_like(batch_dev["stage_idx"])
                output = m(**batch_dev)
                preds_list.append(output.cpu().numpy())
        preds = np.concatenate(preds_list)
        all_preds_seqonly.append(preds)
    ensemble_preds_seqonly = np.mean(all_preds_seqonly, axis=0)

    auc_seqonly = roc_auc_score(all_labels, ensemble_preds_seqonly)
    pr_auc_seqonly = average_precision_score(all_labels, ensemble_preds_seqonly)
    f1_seqonly = f1_score(all_labels, (ensemble_preds_seqonly >= thresh).astype(int))
    mcc_seqonly = matthews_corrcoef(all_labels, (ensemble_preds_seqonly >= thresh).astype(int))
    ba_seqonly = balanced_accuracy_score(all_labels, (ensemble_preds_seqonly >= thresh).astype(int))
    print(f"  AUC={auc_seqonly:.4f}, PR-AUC={pr_auc_seqonly:.4f}, F1={f1_seqonly:.4f}, "
          f"MCC={mcc_seqonly:.4f}, BA={ba_seqonly:.4f}")

    # ── Sequence-only LogReg Baseline ────────────────────────────────────────
    print("\n--- Training Sequence-only LogReg Baseline ---")
    seq_feature_cols = (feature_groups["binned"] + feature_groups["relative_position"] +
                        feature_groups["first_position"] + feature_groups["last_position"] +
                        feature_groups["density_gradient"] + feature_groups["global_count"] +
                        feature_groups["cooccurrence"] + feature_groups["seq_stats"])
    y_lr, preds_lr = train_logistic_baseline(train_df, test_df, seq_feature_cols)
    auc_lr = roc_auc_score(y_lr, preds_lr)
    pr_auc_lr = average_precision_score(y_lr, preds_lr)
    f1_lr = f1_score(y_lr, (preds_lr >= thresh).astype(int))
    mcc_lr = matthews_corrcoef(y_lr, (preds_lr >= thresh).astype(int))
    ba_lr = balanced_accuracy_score(y_lr, (preds_lr >= thresh).astype(int))
    print(f"  AUC={auc_lr:.4f}, PR-AUC={pr_auc_lr:.4f}, F1={f1_lr:.4f}, "
          f"MCC={mcc_lr:.4f}, BA={ba_lr:.4f}")

    # ── DeLong tests ─────────────────────────────────────────────────────────
    print("\n--- DeLong Tests ---")
    p_carcm_vs_seqonly = delong_pvalue(all_labels, ensemble_preds_full, ensemble_preds_seqonly)
    p_carcm_vs_lr = delong_pvalue(all_labels, ensemble_preds_full, preds_lr)
    print(f"  CARCM vs Seq-only (zeroed context): p = {p_carcm_vs_seqonly:.6f}")
    print(f"  CARCM vs Seq-only LogReg:           p = {p_carcm_vs_lr:.6f}")

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "CARCM (full)": {
            "AUC": auc_full, "PR-AUC": pr_auc_full, "F1": f1_full,
            "MCC": mcc_full, "Balanced_Acc": ba_full
        },
        "Seq-only (context zeroed)": {
            "AUC": auc_seqonly, "PR-AUC": pr_auc_seqonly, "F1": f1_seqonly,
            "MCC": mcc_seqonly, "Balanced_Acc": ba_seqonly
        },
        "Seq-only LogReg": {
            "AUC": auc_lr, "PR-AUC": pr_auc_lr, "F1": f1_lr,
            "MCC": mcc_lr, "Balanced_Acc": ba_lr
        },
        "DeLong p-values": {
            "CARCM_vs_SeqOnly_zeroed": float(p_carcm_vs_seqonly),
            "CARCM_vs_SeqOnly_LogReg": float(p_carcm_vs_lr),
        }
    }

    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for model_name, metrics in results.items():
        if model_name.startswith("DeLong"):
            continue
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)

    # ── Visualization ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Context Ablation Ladder — Homology-Aware Split",
                 fontsize=14, fontweight="bold")

    # (a) AUC comparison
    ax = axes[0]
    model_names = ["CARCM\n(full)", "Seq-only\n(zeroed ctx)", "Seq-only\nLogReg"]
    aucs = [auc_full, auc_seqonly, auc_lr]
    colors = ["#4CAF50", "#FF9800", "#2196F3"]
    bars = ax.bar(model_names, aucs, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("(a) AUC Comparison", fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # (b) Multi-metric comparison
    ax = axes[1]
    metrics_names = ["AUC", "PR-AUC", "F1", "MCC", "BA"]
    x = np.arange(len(metrics_names))
    width = 0.25
    for i, (mname, col) in enumerate(zip(["CARCM (full)", "Seq-only (context zeroed)", "Seq-only LogReg"], colors)):
        vals = [results[mname].get(m, 0) for m in ["AUC", "PR-AUC", "F1", "MCC", "Balanced_Acc"]]
        ax.bar(x + i*width, vals, width, label=mname, color=col)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Score")
    ax.set_title("(b) Multi-Metric Comparison", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    # (c) ΔAUC from random vs homology split
    ax = axes[2]
    random_auc_file = "/home/saiful/CARCM/models_carcm/best/best_results.json"
    random_auc = None
    if os.path.exists(random_auc_file):
        with open(random_auc_file) as f:
            rdata = json.load(f)
            random_auc = rdata.get("metrics", {}).get("test_auc", None)

    if random_auc is not None:
        labels_x = ["Random\nSplit", "Homology\nSplit"]
        vals_x = [random_auc, auc_full]
        bars = ax.bar(labels_x, vals_x, color=["#9C27B0", "#4CAF50"],
                       edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals_x):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        delta = random_auc - auc_full
        ax.set_title(f"(c) Random vs Homology Split\n(ΔAUC = {delta:.4f})",
                     fontweight="bold")
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("AUC")
    else:
        ax.text(0.5, 0.5, "Random split\nresults not found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) Random vs Homology Split", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_ladder.png"), dpi=400, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_ladder.pdf"), bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
