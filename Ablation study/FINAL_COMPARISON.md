# CARCM Ablation Study — Final Comparison

This document compares results across **four complementary ablation studies**
that together pin down what makes CARCM work. All studies share:

- The same fixed test set (random-split test, n = 4,451)
- The same training regime (LR = 1e-3, hidden = 128, dropout = 0.4, 10 seeds,
  150 epochs, patience 25 — matching the v7c headline regime)
- The same v7b-style 5-method ensemble selection (best of simple-mean /
  weighted-avg / rank-avg / meta-learner / blended)
- Significance assessed by paired DeLong tests on ensemble predictions

The 4 studies, in increasing-detail order:

| # | Study | Question answered | Configs |
|---|---|---|---|
| 1 | **Context ablation** | Does the (V, T, S) context branch carry signal? | 8 |
| 2 | **LOO with context** | Within Full CARCM, which sequence-feature block is doing the work? | 4 |
| 3 | **LOO without context** | Are sequence features individually redundant *intrinsically*? | 4 |
| 4 | **LOI with context** | What is each sequence feature's *upper-bound* contribution? | 4 |

Total: 20 configurations × 10 seeds = 200 trained models.

---

## 1. Headline numbers (ensemble AUC under v7b best-of selection)

| # | Configuration                                     | Ensemble AUC |
|---|---------------------------------------------------|--------------|
| 1 | **Full CARCM** (4 sequence features + 3 contexts) | **0.8531**   |
| 2 | Sequence-only (no context)                        | 0.6211       |
| 7 | Context-only (no sequence features)               | 0.6811       |
| 6 | Permuted-label control                            | 0.5977       |
| 8 | Parameter-matched bias control                    | 0.6205       |

The Full row reproduces the v7c headline (~0.852) within rounding. Same
training pipeline, same ensemble logic.

---

## 2. Study 1 — Context ablation (Configs 1–8)

| # | Configuration               | Ensemble AUC | ΔAUC vs Seq-only | DeLong p vs Full |
|---|-----------------------------|--------------|------------------|------------------|
| 1 | Full CARCM                  | **0.8531**   | +0.232           | ref              |
| 2 | Sequence-only               | 0.6211       | 0 (ref)          | <1e-300 ****     |
| 3 | Sequence + Variety only     | 0.6884       | +0.067           | <1e-300 ****     |
| 4 | Sequence + Tissue only      | 0.7614       | +0.140           | <1e-300 ****     |
| 5 | Sequence + Stage only       | 0.6905       | +0.069           | <1e-300 ****     |
| 6 | **Permuted labels**         | **0.5977**   | **−0.023**       | <1e-300 ****     |
| 7 | Context-only                | 0.6811       | +0.060           | <1e-300 ****     |
| 8 | **Param-matched bias**      | **0.6205**   | **−0.001**       | <1e-300 ****     |

**Takeaway.** Context contributes +0.23 AUC. The two negative controls (rows 6
and 8) collapse to ≤ sequence-only AUC, ruling out both "context is noise"
and "the win is just extra parameters". Tissue is the strongest single
context dimension (+0.14), then stage and variety (~+0.07 each).

---

## 3. Study 2 — LOO with context (Configs 9–12)

Each row removes one sequence-side feature block while keeping full context.
Reference is Full CARCM (0.8531).

| # | Configuration       | Ensemble AUC | ΔAUC vs Full | DeLong p vs Full |
|---|---------------------|--------------|--------------|------------------|
| 1 | Full CARCM (ref)    | **0.8531**   | 0 (ref)      | —                |
| 9 | − Binned            | 0.8405       | **−0.013**   | 1.1e-4 ***       |
| 10| − Positional        | 0.8503       | −0.003       | 0.27 n.s.        |
| 11| − Global Counts     | 0.8548       | +0.002       | 0.48 n.s.        |
| 12| − Co-occurrence     | 0.8539       | +0.001       | 0.71 n.s.        |

**Takeaway under context.** Only binned removal moves the ensemble
significantly. The other three are statistically indistinguishable from Full.
*If we stopped here we would conclude that 3 of 4 features are noise* — but
that turns out to be a misleading view. (See Study 4.)

---

## 4. Study 3 — LOO without context (Configs 13–16)

Same LOO, but with the context branch removed first. Reference is
sequence_only (0.6211).

| #  | Configuration             | Ensemble AUC | ΔAUC vs Seq-only | DeLong p vs Seq-only |
|----|---------------------------|--------------|------------------|----------------------|
| 2  | Sequence-only (ref)       | 0.6211       | 0 (ref)          | —                    |
| 13 | Seq-only − Binned         | 0.6252       | −0.006 [−0.014, +0.001] | n.s. (p=0.27) |
| 14 | Seq-only − Positional     | 0.6206       | −0.002 [−0.005, +0.002] | n.s. (p=0.95) |
| 15 | Seq-only − Global Counts  | 0.6216       | +0.001 [−0.002, +0.005] | n.s. (p=0.86) |
| 16 | Seq-only − Co-occurrence  | 0.6199       | −0.003 [−0.005, +0.000] | n.s. (p=0.87) |

**Takeaway without context.** The same picture: removing any single feature
block changes ensemble AUC by ≤ 0.006, and none clear the 95% CI bar.
Studies 2 and 3 together show that LOO answers a question about **redundancy
inside the trained model**, not about each feature's intrinsic signal.

---

## 5. Study 4 — LOI with context (Configs 17–20) — the new view

Each row keeps **only one** sequence feature block plus full context. Two
references: Full CARCM (Δ = how much each LOI loses) and context_only (Δ =
how much each LOI adds on top of pure context).

| #  | Configuration                       | Ensemble AUC | ΔAUC vs Full | ΔAUC vs context_only | DeLong p vs Full | % of Full's gain over ctx-only |
|----|-------------------------------------|--------------|--------------|----------------------|------------------|--------------------------------|
| 1  | Full CARCM (ref)                    | **0.8531**   | 0 (ref)      | +0.1720              | —                | 100%                           |
| 17 | **Context + Binned only**           | **0.8484**   | −0.005       | **+0.1336**          | 0.105 n.s.       | **97%**                        |
| 19 | Context + Global Counts only        | 0.8389       | −0.014       | +0.1175              | 7.4e-4 ***       | 84%                            |
| 18 | Context + Positional only           | 0.8135       | −0.040       | +0.1043              | <1e-300 ****     | 76%                            |
| 20 | Context + Co-occurrence only        | 0.7569       | −0.096       | +0.0561              | <1e-300 ****     | 41%                            |
| 7  | Context-only (ref)                  | 0.6811       | −0.172       | 0 (ref)              | <1e-300 ****     | 0%                             |
| 2  | Sequence-only (no context)          | 0.6211       | −0.232       | −0.0600              | <1e-300 ****     | —                              |

**Takeaway from LOI.** This is where the previously-hidden signal becomes
visible:

- **Binned alone is essentially tied with Full** (Δ vs Full = −0.005, n.s.)
  and recovers 97% of Full's gain over context_only. Binned motif counts is
  the load-bearing sequence feature.
- **Global counts alone reaches 0.839** — only 0.014 below Full and 84% of
  the gain over context-only. Count is a strong feature too, just redundant
  with binned (since `count = Σ_b binned[m,b]`).
- **Positional alone reaches 0.814** — 76% of the gain. Positional features
  carry real information.
- **Co-occurrence alone reaches 0.757** — only 41% of the gain. Cooc is the
  weakest sequence-side feature, but still adds +0.056 over context-only.

The four single-feature ceilings span a 0.092 AUC range (0.757 → 0.848). Each
feature individually can pull a substantial fraction of the headline gain
when combined with context.

---

## 6. Why LOO and LOI tell different stories — and which is "right"

| Question                                                        | LOO answer                          | LOI answer                          |
|-----------------------------------------------------------------|-------------------------------------|-------------------------------------|
| Is feature *X* redundant in the trained Full model?             | **Yes** for positional/count/cooc   | (not what LOI measures)             |
| What is feature *X*'s intrinsic contribution under context?     | (not what LOO measures)             | **Substantial: +0.06 to +0.13**     |
| Are the four features overlapping with each other?              | **Yes** (LOO shows redundancy)      | **Yes** (LOI ceilings are similar)  |
| Is binned the most important?                                   | **Yes** (only LOO drop is significant) | **Yes** (only_binned closest to Full) |
| Could we train a simpler model with fewer feature blocks?       | **Yes** (LOO suggests we'd lose nothing) | **Yes** (binned alone is near Full) |

**Both ablations point to the same conclusion** — the four sequence-side
feature blocks are highly overlapping, and binned motif counts dominates —
but only LOI exposes the *magnitude* of each feature's individual ceiling.
LOO alone would have understated each feature's contribution because
redundancy with the other three masks it.

For the paper, **report both LOO and LOI side-by-side**: LOO answers
"how robust is the trained model to losing one block?" and LOI answers
"how much can each block contribute on its own?". Together they tell a
complete story.

---

## 7. The single most important plot

**`feature_loi/figures/loi_delta_auc.png`** — two-panel comparison:
- Left: ΔAUC vs Full (single-feature drop from the Full model)
- Right: ΔAUC vs context_only (single-feature gain over pure context)

This figure makes the LOI hierarchy immediately readable: Binned ≈ Full;
Count ≈ 84% of gain; Positional ≈ 76%; Cooc ≈ 41%. None of these
contributions show up in the LOO view because the other three features
substitute for any one removed.

---

## 8. Reviewer-defensible claims (final)

1. **Context is load-bearing.** Permuted-label control collapses below
   sequence-only; parameter-matched bias control matches sequence-only.
   The +0.23 AUC gain over sequence-only is therefore real signal, not
   noise or capacity.

2. **The four sequence features are overlapping but each is individually
   informative.** LOI shows that any single sequence feature paired with
   context recovers between 41% (cooc) and 97% (binned) of the headline
   gain over context-only. They are *not* redundant in the sense of "no
   signal"; they are redundant in the sense of "highly correlated views of
   the same underlying motif occurrences".

3. **Binned motif counts is the dominant sequence representation.** Removal
   under full context drops AUC significantly (LOO ***); inclusion alone
   with context recovers the full model's performance (LOI n.s. vs Full).
   No other feature has both properties.

4. **Global counts is a strict marginal of binned and is mostly redundant.**
   `count[m] = Σ_b binned[m, b]`, so the LSTM-on-counts path can only add
   per-position information that binned doesn't already encode. LOO removal
   under full context yields a slight (per-seed) gain, indicating mild
   overfitting; LOI alone with context still reaches 0.839, suggesting
   counts inherit most of the predictive signal of binned through the
   marginal relationship.

5. **A simpler "binned + context" model would match Full at this
   training regime.** only_binned + context = 0.8484 vs Full = 0.8531
   (Δ = −0.005, n.s.). For runtime-constrained deployment or follow-up
   work, this is a defensible architectural simplification.

---

## 9. File index — final layout

### Parent study (`Ablation study/`)

| Path | Contents |
|---|---|
| `results/aggregate_metrics.csv`                          | All 16 context + LOO + seq-only LOO configs |
| `results/per_seed_long.csv`                              | Long-format per-seed metrics (160 rows) |
| `results/delong_full_pairwise.csv`                       | DeLong z + p (vs Full and vs Seq-only) |
| `results/ensemble_selection.csv`                         | Best ensemble method per config |
| `figures/ablation_delta_auc.{png,pdf}`                   | Context ablation ΔAUC bar chart |
| `figures/ablation_metric_grid.{png,pdf}`                 | Context AUC/PR-AUC/F1/MCC grid |
| `figures/feature_ablation_delta_auc.{png,pdf}`           | LOO with context ΔAUC |
| `figures/feature_seqonly_delta_auc.{png,pdf}`            | LOO without context ΔAUC |
| `figures/feature_loo_compare.{png,pdf}`                  | Side-by-side LOO with/without context |
| `report/ablation_report.{md,docx}`                       | Auto-generated narrative report |
| `SUMMARY.md`                                             | Earlier (3-study) human-readable summary |
| `FINAL_COMPARISON.md`                                    | **This document** |
| `ablation_model.py`, `ablation_train.py`, …              | Self-contained study code |

### LOI study (`Ablation study/feature_loi/`)

| Path | Contents |
|---|---|
| `results/loi_aggregate_metrics.csv`                      | LOI table (4 configs + 3 references) |
| `results/loi_per_seed_long.csv`                          | Long-format per-seed metrics |
| `results/loi_delong.csv`                                 | DeLong z + p vs Full and vs context_only |
| `results/loi_ensemble_selection.csv`                     | Best ensemble method per LOI config |
| `figures/loi_delta_auc.{png,pdf}`                        | **Two-panel LOI bar chart** (the headline figure) |
| `figures/loi_metric_grid.{png,pdf}`                      | LOI 4-metric grid |
| `report/loi_report.{md,docx}`                            | Auto-generated LOI narrative |
| `loi_configs.py`, `loi_train.py`, `loi_aggregate.py`, …  | Self-contained LOI code (imports parent's model + utils) |

---

## 10. One-paragraph summary for the paper abstract / intro

> A four-part ablation study (20 configurations × 10 seeds) decomposes
> CARCM's performance gains. The (Variety, Tissue, Stage) context branch
> contributes ΔAUC = +0.23 over a sequence-only baseline (DeLong p < 1e-300),
> and is robust to two negative controls: random label permutation drops AUC
> below sequence-only, and a parameter-matched learnable bias matches
> sequence-only — ruling out both "context is noise" and "the win is just
> extra parameters". Within the sequence side, the four motif-feature blocks
> (binned counts, positional features, global counts, co-occurrence) are
> highly overlapping: leave-one-out removes only binned counts significantly
> (−0.013, p < 1e-3) and the others not at all. Leave-one-in reveals that
> each block individually carries substantial signal — binned alone with
> context recovers 97% of the headline gain (AUC = 0.848 vs Full's 0.853,
> n.s.); global counts, positional, and co-occurrence individually recover
> 84%, 76%, and 41% respectively. The four feature blocks are therefore
> redundant in the sense of "highly correlated views of the same underlying
> motif occurrences" rather than redundant in the sense of "no signal".
