# Ablation Study — Paper Presentation Plan (Memory File)

This file records, for future reference, exactly which of the 20
configurations trained inside this directory should appear in the paper,
in which form, and where (main text vs supplement).

---

## 1. What's in this directory

20 trained configurations × 10 seeds, split into 4 logical sub-studies:

| Sub-study | Configs | Source |
|---|---|---|
| **A. Context ablation** | 8 | `results/{full, sequence_only, seq_variety, seq_tissue, seq_stage, permuted_labels, context_only, param_matched_bias}/` |
| **B. Feature LOO under full context** | 4 | `results/{no_binned, no_positional, no_count, no_cooc}/` |
| **C. Feature LOO under no context** | 4 | `results/{seq_only_no_binned, seq_only_no_positional, seq_only_no_count, seq_only_no_cooc}/` |
| **D. Feature LOI under full context** | 4 | `feature_loi/results/{only_binned, only_positional, only_count, only_cooc}/` |

---

## 2. Paper plan: 2 tables, 3 figures

20 rows in one table will overwhelm reviewers. 20 rows split into 4 separate
small tables looks like padding. The clean middle ground is:

- **1 main-text table + 1 supplementary table**
- **1 main-text figure + 2 supplementary figures**

---

## 3. Main paper — Table 2 (architecture ablation, 8 rows)

Pick the rows that each answer a distinct reviewer question. Eight rows is
the sweet spot.

| # | Configuration                       | Ensemble AUC | ΔAUC vs Seq-only | DeLong p vs Full | Why this row is in the table              |
|---|-------------------------------------|--------------|------------------|------------------|-------------------------------------------|
| 1 | **Full CARCM**                      | **0.8531**   | +0.232           | reference        | the model                                 |
| 2 | Sequence-only                       | 0.6211       | 0 (ref)          | <1e-300 ****     | defines the gain to explain               |
| 3 | Context-only                        | 0.6811       | +0.060           | <1e-300 ****     | upper bound on label-prior memorisation   |
| 4 | **Permuted-label control**          | **0.5977**   | **−0.023**       | <1e-300 ****     | proves context is real signal             |
| 5 | **Param-matched bias control**      | **0.6205**   | **−0.001**       | <1e-300 ****     | rules out the parameter-count confound    |
| 6 | Sequence + Tissue only              | 0.7614       | +0.140           | <1e-300 ****     | strongest single context dimension        |
| 7 | **only_binned + context**           | **0.8484**   | +0.227           | 0.105 n.s.       | binned alone matches Full                 |
| 8 | only_cooc + context                 | 0.7569       | +0.136           | <1e-300 ****     | weakest single feature (range of LOI)     |

**Why this set:** rows 1–5 settle the "context is real / not just parameters"
debate; row 6 names the strongest context dimension; rows 7–8 frame the LOI
ceiling range (best to worst single-feature). Eight rows answer every
objection a reviewer is likely to raise.

---

## 4. Main paper — Figure 1 (one figure)

Use the existing `figures/ablation_delta_auc.png` — it already shows context
ablation rows 1–7 with paired-seed 95% CIs and DeLong asterisks at 400 dpi.
No re-rendering needed.

---

## 5. Supplementary Table S1 — Full feature decomposition (16 rows)

The single supplementary table that bundles all 12 feature-level rows from
sub-studies B + C + D. Three groups of 4 rows, plus references at the top.

| # | Configuration                         | Ensemble AUC | ΔAUC vs reference | reference   | Significance         |
|---|---------------------------------------|--------------|-------------------|-------------|----------------------|
|   | *Reference rows*                      |              |                   |             |                      |
| 1 | Full CARCM                            | 0.8531       | —                 | —           | —                    |
| 2 | Sequence-only                         | 0.6211       | —                 | —           | —                    |
| 7 | Context-only                          | 0.6811       | —                 | —           | —                    |
|   | *Group B — LOO with context (vs Full)*|              |                   |             |                      |
| 9 | − Binned                              | 0.8405       | −0.013            | Full        | DeLong p=1.1e-4 ***  |
| 10| − Positional                          | 0.8503       | −0.003            | Full        | n.s.                 |
| 11| − Global Counts                       | 0.8548       | +0.002            | Full        | n.s.                 |
| 12| − Co-occurrence                       | 0.8539       | +0.001            | Full        | n.s.                 |
|   | *Group C — LOO without context (vs Sequence-only)* |  |                |             |                      |
| 13| − Binned (no ctx)                     | 0.6252       | +0.004            | Seq-only    | n.s.                 |
| 14| − Positional (no ctx)                 | 0.6206       | −0.001            | Seq-only    | n.s.                 |
| 15| − Global Counts (no ctx)              | 0.6216       | +0.001            | Seq-only    | n.s.                 |
| 16| − Co-occurrence (no ctx)              | 0.6199       | −0.001            | Seq-only    | n.s.                 |
|   | *Group D — LOI with context (vs Full)*|              |                   |             |                      |
| 17| only_binned + ctx                     | 0.8484       | −0.005            | Full        | n.s.                 |
| 19| only_count + ctx                      | 0.8389       | −0.014            | Full        | DeLong p=7.4e-4 ***  |
| 18| only_positional + ctx                 | 0.8135       | −0.040            | Full        | <1e-300 ****         |
| 20| only_cooc + ctx                       | 0.7569       | −0.096            | Full        | <1e-300 ****         |

This single table tells the complete feature-level story. **Crucially, both
LOO and LOI must be present** — LOO alone makes 3 of 4 features look like
noise; LOI alone hides the redundancy. The pair is what's defensible.

---

## 6. Supplementary Figures

| # | Figure | Path | Why |
|---|---|---|---|
| S-F1 | LOI two-panel | `feature_loi/figures/loi_delta_auc.png` | Cleanest single visual for the feature-level story |
| S-F2 | LOO with-vs-without context | `figures/feature_loo_compare.png` | Shows redundancy persists even with the gating fallback removed |

Both already exist at 400 dpi.

---

## 7. What to omit from the paper

| Configs                                            | Why omit                                                                 |
|----------------------------------------------------|--------------------------------------------------------------------------|
| Sequence + Variety only (config 3)                  | Variety contribution (+0.07) is significant but redundant with Tissue row in main paper. Belongs in supplement S1 if expanded. |
| Sequence + Stage only (config 5)                    | Same logic                                                               |
| `ablation_metric_grid.png` (8-config 4-metric grid) | Metric grids are noisy and largely redundant with ΔAUC bar chart        |
| `feature_ablation_metric_grid.png`, `feature_seqonly_delta_auc.png`, `loi_metric_grid.png` | Bundle metrics into table cells; keep figures focused on ΔAUC          |
| All `console_outputs/` logs                         | Supplementary data deposit, not paper content                            |

---

## 8. Summary — what to put where

| Paper location           | Item                                          | Source                                               |
|--------------------------|-----------------------------------------------|------------------------------------------------------|
| **Main text — Table 2**  | Architecture ablation (8 rows)                | `aggregate_metrics.csv` + `feature_loi/results/loi_aggregate_metrics.csv` |
| **Main text — Figure 1** | Context ablation ΔAUC bar chart               | `figures/ablation_delta_auc.png`                     |
| **Supplement — Table S1**| Full feature decomposition (16 rows)          | combined CSV                                         |
| **Supplement — Figure S1**| LOI two-panel                                | `feature_loi/figures/loi_delta_auc.png`              |
| **Supplement — Figure S2**| LOO with-vs-without context                  | `figures/feature_loo_compare.png`                    |

That's it: **two tables, three figures**, drawn entirely from this
directory. The 8-row main-text table answers the high-stakes reviewer
questions; the 16-row supplementary table backs up every claim; both
figures are pre-rendered at Q1-acceptable 400 dpi.
