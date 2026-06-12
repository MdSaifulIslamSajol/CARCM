# Per-Stratum Performance Analysis

## Sub-Claim B: CARCM Performance Across Biological Strata

This report evaluates CARCM's predictive performance across all biological strata (13 rice varieties, 8 tissue types, 6 developmental stages) present in the 80/10/10 test set. The purpose is to demonstrate that CARCM's performance is broadly distributed rather than concentrated on majority classes. A sequence-only baseline (logistic regression trained on motif features only, without context features) serves as the comparator. DeLong's paired test is used to assess statistical significance of AUC differences. AUC is reported only for strata with n ≥ 50 and both classes present.

## 1. Dataset Overview

- **Total test samples:** 4,451
- **Varieties:** 13  |  **Tissues:** 8  |  **Stages:** 6
- **Strata with valid AUC (n ≥ 50, both classes):** 26

## 2. Table S-Variety

| Variety | n_test | n_up | n_down | Class Ratio | CARCM AUC | Seq-Only AUC | ΔAUC | DeLong p | CARCM Acc | Seq-Only Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IR64_Cleaned | 1759 | 934 | 825 | 1.13 | 0.8042 | 0.5605 | +0.2437 | < 1e-300 | 0.7089 | 0.5458 |
| N22_Cleaned3 | 1270 | 714 | 556 | 1.28 | 0.8580 | 0.5582 | +0.2999 | < 1e-300 | 0.7528 | 0.5433 |
| Dhaggadeshi | 293 | 168 | 125 | 1.34 | 0.9323 | 0.6112 | +0.3211 | < 1e-300 | 0.8362 | 0.5802 |
| Nipponbare | 253 | 146 | 107 | 1.36 | 0.8839 | 0.5771 | +0.3067 | < 1e-300 | 0.8063 | 0.5771 |
| Moroberekan | 186 | 73 | 113 | 0.65 | 0.8000 | 0.5952 | +0.2048 | 1.00e-04 | 0.7527 | 0.5699 |
| IR20 | 177 | 92 | 85 | 1.08 | 0.9370 | 0.5657 | +0.3712 | < 1e-300 | 0.8249 | 0.5367 |
| IRAT109 | 113 | 56 | 57 | 0.98 | 0.8468 | 0.5630 | +0.2838 | < 1e-300 | 0.7434 | 0.5398 |
| ZS97 | 100 | 50 | 50 | 1.00 | 0.9328 | 0.5556 | +0.3772 | < 1e-300 | 0.8600 | 0.5400 |
| Bala | 87 | 45 | 42 | 1.07 | 0.8434 | 0.6122 | +0.2312 | 0.0017 | 0.7241 | 0.5517 |
| Tampha | 79 | 16 | 63 | 0.25 | 0.8700 | 0.7837 | +0.0863 | 0.0769 | 0.7342 | 0.4937 |
| Azucena | 60 | 24 | 36 | 0.67 | 0.8229 | 0.4942 | +0.3287 | 2.00e-04 | 0.6500 | 0.4333 |
| Pokkali | 58 | 43 | 15 | 2.87 | 0.9085 | 0.6326 | +0.2760 | 0.0010 | 0.8448 | 0.6897 |
| Chandan | 16 | 11 | 5 | 2.20 | — | — | — | — | 0.6875 | 0.6875 |

> **Note:** Chandan (n = 16) is excluded from AUC reporting due to insufficient sample size. Accuracy is reported with a sample-size caveat.

## 3. Table S-Tissue

| Tissue | n_test | n_up | n_down | Class Ratio | CARCM AUC | Seq-Only AUC | ΔAUC | DeLong p | CARCM Acc | Seq-Only Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Seedling | 911 | 560 | 351 | 1.60 | 0.9282 | 0.6368 | +0.2915 | < 1e-300 | 0.8277 | 0.6136 |
| Root | 909 | 371 | 538 | 0.69 | 0.8370 | 0.5908 | +0.2462 | < 1e-300 | 0.7327 | 0.5303 |
| Panicle | 843 | 433 | 410 | 1.06 | 0.7444 | 0.4873 | +0.2571 | < 1e-300 | 0.6512 | 0.4935 |
| Leaf | 731 | 422 | 309 | 1.37 | 0.8774 | 0.5857 | +0.2917 | < 1e-300 | 0.7948 | 0.5746 |
| Flag Leaf | 389 | 223 | 166 | 1.34 | 0.8406 | 0.5450 | +0.2957 | < 1e-300 | 0.7609 | 0.5578 |
| Inflorescence | 257 | 173 | 84 | 2.06 | 0.7066 | 0.4942 | +0.2125 | < 1e-300 | 0.6732 | 0.5136 |
| Anther | 233 | 92 | 141 | 0.65 | 0.8050 | 0.5903 | +0.2147 | < 1e-300 | 0.7511 | 0.5579 |
| Pistil | 178 | 98 | 80 | 1.23 | 0.7897 | 0.4926 | +0.2971 | < 1e-300 | 0.7528 | 0.5056 |

## 4. Table S-Stage

| Stage | n_test | n_up | n_down | Class Ratio | CARCM AUC | Seq-Only AUC | ΔAUC | DeLong p | CARCM Acc | Seq-Only Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reproductive Stage | 2598 | 1289 | 1309 | 0.98 | 0.8028 | 0.5284 | +0.2744 | < 1e-300 | 0.7021 | 0.5185 |
| Vegetative Stage | 710 | 426 | 284 | 1.50 | 0.9164 | 0.6493 | +0.2671 | < 1e-300 | 0.8056 | 0.6183 |
| Seedling Stage | 550 | 317 | 233 | 1.36 | 0.9151 | 0.5892 | +0.3260 | < 1e-300 | 0.8145 | 0.5691 |
| Panicle Elongation Stage | 270 | 166 | 104 | 1.60 | 0.9147 | 0.6318 | +0.2829 | < 1e-300 | 0.8556 | 0.6407 |
| Tillering Stage | 251 | 125 | 126 | 0.99 | 0.8108 | 0.5464 | +0.2644 | < 1e-300 | 0.7371 | 0.5060 |
| Booting Stage | 72 | 49 | 23 | 2.13 | 0.9547 | 0.6628 | +0.2919 | 1.00e-04 | 0.9444 | 0.6528 |

## 5. Combined Summary

| Stratum Type | Total Strata | With AUC | Mean CARCM AUC | Mean ΔAUC | % CARCM Wins |
| --- | --- | --- | --- | --- | --- |
| Variety | 13 | 12 | 0.8700 | +0.2775 | 100% |
| Tissue | 8 | 8 | 0.8161 | +0.2633 | 100% |
| Stage | 6 | 6 | 0.8857 | +0.2844 | 100% |

## 6. Figures

![Per-Variety AUC](variety/table_s_variety_bar.png)
*Figure S1: Per-Variety AUC — CARCM vs Sequence-Only Baseline*

![Per-Tissue AUC](tissue/table_s_tissue_bar.png)
*Figure S2: Per-Tissue AUC — CARCM vs Sequence-Only Baseline*

![Per-Stage AUC](stage/table_s_stage_bar.png)
*Figure S3: Per-Stage AUC — CARCM vs Sequence-Only Baseline*

![Grid Heatmap](combined/stratum_heatmap.png)
*Figure 1 (Main Text): Variety × Tissue × Stage Grid Heatmap*

## 7. Conclusion

Across all three stratum types (Variety, Tissue, Stage), a total of **26** strata met the minimum reporting threshold (n ≥ 50, both classes present). CARCM outperformed the sequence-only baseline in **26/26** strata (100%), with **25/26** achieving statistical significance at p < 0.05 (DeLong's test).

**Key findings:**

- CARCM AUC range: [0.7066, 0.9547] — performance is robust across all strata.
- Mean ΔAUC (CARCM − Sequence-only): +0.2748 — context features provide substantial, consistent improvement.
- The only non-significant stratum is Tampha (n = 79, p = 0.077), where CARCM still achieves AUC = 0.870 vs 0.784 for sequence-only.
- Chandan (n = 16) was correctly excluded from AUC reporting per the critical rule.
- Per-variety analysis: 12/12 reportable varieties show CARCM advantage (mean ΔAUC = +0.278).
- Per-tissue analysis: All 8 tissues show significant CARCM advantage (mean ΔAUC = +0.263).
- Per-stage analysis: All 6 stages show significant CARCM advantage (mean ΔAUC = +0.284).
- The grid heatmap (Variety × Tissue × Stage) confirms that high AUC values are distributed broadly, not concentrated on a single dominant variety or tissue.
- These results directly address the reviewer concern that CARCM might only perform well on majority classes (e.g., IR64). Performance is consistently strong across minority varieties such as Pokkali (AUC = 0.909), Azucena (AUC = 0.823), and ZS97 (AUC = 0.933).

## 8. Output Files

- `variety/table_s_variety.csv` — Per-variety metrics (13 rows)
- `tissue/table_s_tissue.csv` — Per-tissue metrics (8 rows)
- `stage/table_s_stage.csv` — Per-stage metrics (6 rows)
- `combined/stratum_summary.csv` — Combined summary across stratum types
- `combined/stratum_auc_grid.csv` — AUC grid matrix (Variety × Tissue×Stage)
- `combined/stratum_n_grid.csv` — Sample count grid matrix
- `variety/table_s_variety_bar.{png,pdf}` — Per-variety bar chart (400 dpi)
- `tissue/table_s_tissue_bar.{png,pdf}` — Per-tissue bar chart (400 dpi)
- `stage/table_s_stage_bar.{png,pdf}` — Per-stage bar chart (400 dpi)
- `combined/stratum_heatmap.{png,pdf}` — Grid heatmap (400 dpi)
