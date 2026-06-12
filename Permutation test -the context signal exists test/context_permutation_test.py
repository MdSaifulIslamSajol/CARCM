#!/usr/bin/env python3
"""
Context Permutation Test — "The Context Signal Exists" Test
============================================================
At test time, replace each sample's context one-hot with a randomly
chosen other sample's context. If AUC drops, context carries real signal.

Distinct from ablation row #6 (permuted-label training):
  - Row #6: trains on permuted labels → tests if model *needs* real context to learn
  - This test: trains normally, permutes only at inference → measures how much
    context matters at prediction time

1000 permutation iterations. No retraining.
"""

import os
import sys
import json
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/saiful/CARCM/output/"
CHECKPOINT_DIR = "/media/exxact1/saiful/DeepRice/Checkpoints/binbased_carcm/"
OUT_DIR = "/home/saiful/CARCM/Permutation test -the context signal exists test"
RUN_NAME = "best"
NUM_SEEDS = 10
N_PERMUTATIONS = 1000

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "figure.dpi": 400,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

os.makedirs(OUT_DIR, exist_ok=True)

# ── Import model classes from rice_classifier_v7b ────────────────────────────
sys.path.insert(0, "/home/saiful/CARCM")
from rice_classifier_v7b import (
    ContextEnhancedGatedFusionV5,
    PositionalMotifDataset,
    create_context_interactions,
    identify_hard_examples,
    load_feature_groups,
    NUM_BINS,
    set_seed,
)


def load_all_models(device, n_motifs, n_bins, n_cooc,
                    n_varieties, n_tissues, n_stages,
                    hidden_dim, dropout, n_vt, n_ts, n_vs):
    """Load all 10 ensemble checkpoints."""
    models = []
    base_seeds = [42 + i * 1000 for i in range(NUM_SEEDS)]

    for seed in base_seeds:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_seed{seed}.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = ContextEnhancedGatedFusionV5(
            n_motifs=n_motifs, n_bins=n_bins, n_cooc=n_cooc,
            n_varieties=n_varieties, n_tissues=n_tissues, n_stages=n_stages,
            num_seq_stats=4, hidden_dim=hidden_dim,
            embed_dim=64, dropout=dropout,
            n_vt=n_vt, n_ts=n_ts, n_vs=n_vs
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    return models


def get_ensemble_predictions(models, loader, device):
    """Get simple-average ensemble predictions."""
    all_model_preds = []

    for model in models:
        preds = []
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                outputs = model(**batch)
                preds.extend(outputs.cpu().numpy())
        all_model_preds.append(np.array(preds))

    return np.mean(all_model_preds, axis=0)


class PermutedContextDataset(Dataset):
    """Wraps an existing dataset but shuffles context features."""

    def __init__(self, base_dataset, perm_idx):
        self.base = base_dataset
        self.perm_idx = perm_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]

        ctx_idx = self.perm_idx[idx]
        ctx_item = self.base[ctx_idx]

        item["variety"] = ctx_item["variety"]
        item["tissue"] = ctx_item["tissue"]
        item["stage"] = ctx_item["stage"]

        if "variety_tissue" in item:
            item["variety_tissue"] = ctx_item["variety_tissue"]
            item["tissue_stage"] = ctx_item["tissue_stage"]
            item["variety_stage"] = ctx_item["variety_stage"]
            item["variety_idx"] = ctx_item["variety_idx"]
            item["tissue_idx"] = ctx_item["tissue_idx"]
            item["stage_idx"] = ctx_item["stage_idx"]

        return item


def main():
    print("=" * 75)
    print("CONTEXT PERMUTATION TEST")
    print("\"The Context Signal Exists\" Test")
    print("=" * 75)
    print(f"\nPermutation iterations: {N_PERMUTATIONS}")
    print(f"Ensemble models: {NUM_SEEDS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    feature_groups = load_feature_groups(DATA_DIR)

    variety_cols = [c for c in train_df.columns if c.startswith("Variety_")]
    tissue_cols = [c for c in train_df.columns if c.startswith("Tissue_")]
    stage_cols = [c for c in train_df.columns if c.startswith("Stage_")]

    train_context = create_context_interactions(train_df, variety_cols, tissue_cols, stage_cols)
    test_context = create_context_interactions(test_df, variety_cols, tissue_cols, stage_cols)

    count_cols = feature_groups["global_count"]
    train_hard, test_hard, _ = identify_hard_examples(train_df, test_df, count_cols)

    train_dataset = PositionalMotifDataset(
        train_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=train_context, hard_examples=train_hard, fit_scaler=True
    )
    test_dataset = PositionalMotifDataset(
        test_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=test_context, hard_examples=test_hard, scaler=train_dataset.scaler
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_labels = test_dataset.labels

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading 10 ensemble models...")
    n_motifs = 54
    n_cooc = len(feature_groups["cooccurrence"])

    models = load_all_models(
        device, n_motifs, NUM_BINS, n_cooc,
        len(variety_cols), len(tissue_cols), len(stage_cols),
        hidden_dim=128, dropout=0.4,
        n_vt=train_context["n_vt"],
        n_ts=train_context["n_ts"],
        n_vs=train_context["n_vs"],
    )
    print(f"  Loaded {len(models)} models successfully.")

    # ── Observed AUC (true context) ──────────────────────────────────────────
    print("\nComputing observed AUC (true context pairing)...")
    observed_preds = get_ensemble_predictions(models, test_loader, device)
    observed_auc = roc_auc_score(test_labels, observed_preds)
    print(f"  Observed AUC: {observed_auc:.4f}")

    # ── Permutation iterations ───────────────────────────────────────────────
    print(f"\nRunning {N_PERMUTATIONS} permutation iterations...")
    print("  (each iteration: shuffle context → run 10-model ensemble → compute AUC)")

    n_test = len(test_dataset)
    permuted_aucs = []
    rng = np.random.RandomState(42)

    t_start = time.time()
    for i in range(N_PERMUTATIONS):
        perm_idx = rng.permutation(n_test)

        perm_dataset = PermutedContextDataset(test_dataset, perm_idx)
        perm_loader = DataLoader(perm_dataset, batch_size=128, shuffle=False,
                                 num_workers=4, pin_memory=True)

        perm_preds = get_ensemble_predictions(models, perm_loader, device)
        perm_auc = roc_auc_score(test_labels, perm_preds)
        permuted_aucs.append(perm_auc)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (N_PERMUTATIONS - i - 1) / rate
            print(f"  [{i+1:4d}/{N_PERMUTATIONS}]  AUC={perm_auc:.4f}  "
                  f"mean={np.mean(permuted_aucs):.4f}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    total_time = time.time() - t_start
    permuted_aucs = np.array(permuted_aucs)

    # ── Statistical analysis ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("RESULTS")
    print(f"{'='*75}")

    p_value = (np.sum(permuted_aucs >= observed_auc) + 1) / (N_PERMUTATIONS + 1)
    delta_auc = observed_auc - permuted_aucs.mean()
    effect_size = delta_auc / permuted_aucs.std() if permuted_aucs.std() > 0 else float("inf")

    print(f"\n  Observed AUC (true context):     {observed_auc:.4f}")
    print(f"  Permuted AUC (mean ± std):       {permuted_aucs.mean():.4f} ± {permuted_aucs.std():.4f}")
    print(f"  Permuted AUC (min, max):         [{permuted_aucs.min():.4f}, {permuted_aucs.max():.4f}]")
    print(f"  ΔAUC (observed − permuted mean): {delta_auc:+.4f}")
    print(f"  Effect size (Cohen's d):         {effect_size:.2f}")
    print(f"  Permutation p-value:             {p_value:.6f}")
    print(f"  95% CI of permuted AUC:          [{np.percentile(permuted_aucs, 2.5):.4f}, "
          f"{np.percentile(permuted_aucs, 97.5):.4f}]")
    print(f"  Iterations where perm >= obs:    {np.sum(permuted_aucs >= observed_auc)}/{N_PERMUTATIONS}")
    print(f"  Total computation time:          {total_time/60:.1f} minutes")

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "observed_auc": float(observed_auc),
        "permuted_mean": float(permuted_aucs.mean()),
        "permuted_std": float(permuted_aucs.std()),
        "permuted_min": float(permuted_aucs.min()),
        "permuted_max": float(permuted_aucs.max()),
        "delta_auc": float(delta_auc),
        "effect_size_cohens_d": float(effect_size),
        "p_value": float(p_value),
        "ci_95_lower": float(np.percentile(permuted_aucs, 2.5)),
        "ci_95_upper": float(np.percentile(permuted_aucs, 97.5)),
        "n_permutations": N_PERMUTATIONS,
        "n_perm_gte_observed": int(np.sum(permuted_aucs >= observed_auc)),
        "total_time_minutes": round(total_time / 60, 1),
    }

    with open(os.path.join(OUT_DIR, "permutation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    perm_df = pd.DataFrame({"iteration": range(1, N_PERMUTATIONS + 1),
                             "permuted_auc": permuted_aucs})
    perm_df.to_csv(os.path.join(OUT_DIR, "permuted_auc_values.csv"), index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("GENERATING FIGURES")
    print(f"{'='*75}")

    # Figure 1: Main histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(permuted_aucs, bins=40, color="#457b9d", edgecolor="black",
            linewidth=0.5, alpha=0.85, density=True, label="Permuted AUC distribution")

    ax.axvline(observed_auc, color="#e63946", linewidth=2.5, linestyle="-",
               label=f"Observed AUC = {observed_auc:.4f}", zorder=5)

    ax.axvline(permuted_aucs.mean(), color="#f4a261", linewidth=2, linestyle="--",
               label=f"Permuted mean = {permuted_aucs.mean():.4f}")

    ci_lo = np.percentile(permuted_aucs, 2.5)
    ci_hi = np.percentile(permuted_aucs, 97.5)
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="#f4a261", label="95% CI (permuted)")

    ax.set_xlabel("Test AUC", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(f"Context Permutation Test (n = {N_PERMUTATIONS} iterations)\n"
                 f"ΔAUC = {delta_auc:+.4f}, p = {p_value:.4f}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"permutation_histogram.{ext}"))
    plt.close(fig)
    print("  Saved: permutation_histogram.{png,pdf}")

    # Figure 2: Running mean + convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    running_mean = np.cumsum(permuted_aucs) / np.arange(1, N_PERMUTATIONS + 1)
    ax1.plot(range(1, N_PERMUTATIONS + 1), running_mean, color="#264653", linewidth=1)
    ax1.axhline(observed_auc, color="#e63946", linewidth=2, linestyle="-",
                label=f"Observed = {observed_auc:.4f}")
    ax1.axhline(permuted_aucs.mean(), color="#f4a261", linewidth=1.5, linestyle="--",
                label=f"Final mean = {permuted_aucs.mean():.4f}")
    ax1.set_xlabel("Permutation Iteration", fontsize=12)
    ax1.set_ylabel("Running Mean AUC", fontsize=12)
    ax1.set_title("Convergence of Permuted AUC", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    running_p = np.array([(np.sum(permuted_aucs[:i+1] >= observed_auc) + 1) / (i + 2)
                          for i in range(N_PERMUTATIONS)])
    ax2.plot(range(1, N_PERMUTATIONS + 1), running_p, color="#264653", linewidth=1)
    ax2.axhline(0.05, color="red", linewidth=1, linestyle="--", alpha=0.5, label="α = 0.05")
    ax2.set_xlabel("Permutation Iteration", fontsize=12)
    ax2.set_ylabel("Running p-value", fontsize=12)
    ax2.set_title("Convergence of p-value", fontsize=13, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"permutation_convergence.{ext}"))
    plt.close(fig)
    print("  Saved: permutation_convergence.{png,pdf}")

    # Figure 3: Box + strip plot
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(permuted_aucs, positions=[0], widths=0.4,
                    patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#457b9d")
    bp["boxes"][0].set_alpha(0.6)
    ax.scatter(np.random.normal(0, 0.04, size=min(200, len(permuted_aucs))),
               np.random.choice(permuted_aucs, min(200, len(permuted_aucs)), replace=False),
               color="#457b9d", alpha=0.3, s=10, zorder=2)
    ax.scatter([0.6], [observed_auc], color="#e63946", s=200, marker="*",
               zorder=5, label=f"Observed = {observed_auc:.4f}", edgecolors="black")
    ax.set_xticks([0, 0.6])
    ax.set_xticklabels(["Permuted\nContext", "True\nContext"], fontsize=12)
    ax.set_ylabel("Test AUC", fontsize=13)
    ax.set_title("Observed vs Permuted AUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"permutation_boxplot.{ext}"))
    plt.close(fig)
    print("  Saved: permutation_boxplot.{png,pdf}")

    # ── Conclusion ───────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("CONCLUSION")
    print(f"{'='*75}")

    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = f"significant (p = {p_value:.4f})"
    elif p_value < 0.05:
        significance = f"marginally significant (p = {p_value:.4f})"
    else:
        significance = f"NOT significant (p = {p_value:.4f})"

    print(f"\n  The context permutation test is {significance}.")
    print(f"\n  When context one-hots are randomly shuffled at inference time")
    print(f"  (destroying the true variety/tissue/stage pairing while preserving")
    print(f"  the marginal distribution), AUC drops from {observed_auc:.4f} to")
    print(f"  {permuted_aucs.mean():.4f} (ΔAUC = {delta_auc:+.4f}).")

    if p_value < 0.05:
        print(f"\n  This confirms that context features carry REAL biological signal")
        print(f"  at inference time. The model is not merely memorising motif patterns;")
        print(f"  it actively uses the biological context (variety, tissue, stage)")
        print(f"  to modulate its predictions.")
        print(f"\n  The effect size (Cohen's d = {effect_size:.2f}) indicates a")
        if effect_size > 0.8:
            print(f"  LARGE effect — context is a major contributor to prediction quality.")
        elif effect_size > 0.5:
            print(f"  MEDIUM effect — context meaningfully improves predictions.")
        else:
            print(f"  SMALL effect — context provides a modest but real improvement.")
    else:
        print(f"\n  Context features do NOT carry significant signal at inference time.")

    print(f"\n  Distinct from ablation row #6 (permuted-label training):")
    print(f"    - Row #6 tests whether the model needs real context to LEARN.")
    print(f"    - This test measures how much context matters at INFERENCE.")
    print(f"    Both are reported for completeness.")

    print(f"\n  All results saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
