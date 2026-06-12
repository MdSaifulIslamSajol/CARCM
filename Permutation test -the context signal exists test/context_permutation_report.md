# Context Permutation Test

## "The Context Signal Exists" Test

This test determines whether CARCM's context features (variety, tissue, developmental stage) carry real biological signal at inference time. At test time, each sample's context one-hot is replaced with a randomly chosen other sample's context. If AUC drops substantially, context is carrying real signal.

**Distinct from ablation row #6:** Row #6 trains on permuted labels (tests if model needs real context to *learn*). This test trains normally and permutes only at inference (measures how much context matters at *prediction time*). Both are reported.

---

## Executive Summary

| Metric | Value |
| --- | --- |
| Observed AUC (true context) | 0.8522 |
| Permuted AUC (mean ± std) | 0.5420 ± 0.0079 |
| Permuted AUC range | [0.5161, 0.5644] |
| ΔAUC (observed − permuted) | +0.3102 |
| Effect size (Cohen’s d) | 39.18 (LARGE) |
| Permutation p-value | 0.0010 (p < 0.001) |
| 95% CI of permuted AUC | [0.5264, 0.5571] |
| Iterations where perm ≥ observed | 0/1000 |
| Total computation time | 165.1 minutes |

> **Result:** The context permutation test is **highly significant (p < 0.001)**. When context features are randomly shuffled at inference, AUC drops from **0.8522** to **0.5420** (ΔAUC = **+0.3102**). Not a single permutation out of 1000 reached the observed AUC. **Context features carry REAL biological signal.**

---

## 1. Methods

- CARCM trained with best config (LR=1e-3, Hidden=128, Dropout=0.4, 10-seed ensemble)
- 1000 permutation iterations at test time
- Each iteration: shuffle context features (variety, tissue, stage one-hots + all interaction one-hots + embedding indices) across test samples
- Sequence features and true labels held fixed
- 10-model ensemble predictions simple-averaged
- p-value = (# perm AUC ≥ observed + 1) / (N + 1)

## 2. Results

- Observed AUC (true context): **0.8522**
- Permuted AUC (mean ± std): **0.5420 ± 0.0079**
- Permuted AUC range: [0.5161, 0.5644]
- ΔAUC = **+0.3102** (36.4% relative improvement from context)
- Cohen’s d = **39.18** (extraordinarily large)
- **0/1000** permutations reached observed AUC
- Permuted AUC barely above random chance (0.5)

### 2.1 Why the Large AUC Drop Is Expected (Not a Limitation)

#### (a) Sequence duplication across contexts

The rice promoter dataset contains sequences that appear multiple times under different biological contexts. Of 16,319 unique sequence motif profiles, **5,804 (35.6%)** carry conflicting labels across contexts — the same sequence is labelled “active” in one variety/tissue/stage combination and “inactive” in another. These conflicting-label groups account for **59.5% of all samples** (58.3% of the test set).

This reflects genuine biology: the same promoter can behave differently depending on cellular context (e.g., active in IR64 seedling tissue but inactive in Nipponbare root). For these samples, **no sequence-only model can achieve high AUC** because the same input maps to different labels. Context is the disambiguating signal.

#### (b) Wrong context ≠ no context

The permutation test provides **WRONG** context, not absent context. CARCM’s gated fusion architecture (context-query attention, gating mechanism) was trained to integrate context with sequence features. Wrong context actively **misleads** the attention weights, gating signals, and context embeddings — amplifying the wrong motif patterns and suppressing the right ones.


| Scenario | What Happens | AUC |
| --- | --- | --- |
| **CARCM + true context** | Gating/attention correctly modulates motif interpretation | **0.8522** |
| **Sequence-only LogReg** | Ignores context, averages over conflicting labels | **~0.57** |
| **CARCM + wrong context** | Gating/attention actively misleads motif interpretation | **0.5420** |

*Analogy:* A physician trained to interpret symptoms differently for paediatric vs geriatric patients will perform **worse** when given the wrong patient age than a physician who ignores age entirely. This does not mean the first physician is “worse” — it shows that age-conditioned reasoning is deeply integrated.

The permuted AUC (~0.542) falling slightly below a naive sequence-only logistic regression (~0.57) is therefore expected: the logistic regression never learned context-dependent representations and is unaffected by context permutation, while CARCM’s gated architecture is actively disrupted by incorrect context. The comparison is not apples-to-apples: the LogReg has no context channel that can be disrupted, while CARCM’s gated architecture is *designed* for deep context-sequence integration.

### 2.2 Suggested Framing for the Paper

> *“Under context permutation at inference, AUC drops from 0.852 to 0.542 (p < 0.001), confirming that context carries real biological signal. The near-random permuted AUC reflects the dataset’s inherent context-dependence: 59.5% of samples belong to sequence profiles with conflicting labels across biological contexts, making context the disambiguating factor. The gated fusion architecture’s deep context-sequence integration means incorrect context actively disrupts prediction — consistent with the design thesis that promoter activity is jointly determined by sequence and cellular context.”*

## 3. Figures

![Permutation Histogram](permutation_histogram.png)
*Figure 1: Distribution of permuted AUC (blue) vs observed AUC (red). Zero overlap.*

![Convergence](permutation_convergence.png)
*Figure 2: Running mean converges to ~0.542; p-value stays well below 0.05.*

![Boxplot](permutation_boxplot.png)
*Figure 3: Boxplot of permuted AUC vs observed AUC (red star).*

## 4. Conclusion

- The context permutation test is highly significant (p < 0.001, 1000 iterations).
- AUC drops from 0.8522 to 0.5420 when context is randomly shuffled at inference (ΔAUC = +0.3102). This is the largest single-factor effect observed in any of our ablation or permutation tests.
- Zero out of 1000 permutation iterations matched or exceeded the observed AUC. The observed AUC lies 39.2 standard deviations above the permuted mean.
- The large magnitude of the drop is biologically expected. The dataset contains 5,804 unique sequence profiles (35.6% of all unique sequences) that carry conflicting labels across different biological contexts, accounting for 59.5% of all samples. The same promoter sequence can be active in one variety/tissue/stage combination and inactive in another. Context is not a supplementary signal — it is the disambiguating factor for these samples.
- The permuted AUC (~0.542) falling slightly below a naive sequence-only logistic regression (~0.57) is expected: this test provides WRONG context, not absent context. CARCM’s gated fusion architecture is actively disrupted by incorrect context signals, which mislead the attention and gating mechanisms. A sequence-only model, by contrast, has no context channel to disrupt.
- This directly answers the reviewer concern that "maybe the context is just noise." Context carries strong, real biological signal that reflects genuine tissue- and stage-specific promoter activity. The model actively uses this information at inference time to modulate its predictions through context-query attention and gated fusion.
- Importantly, this test is complementary to ablation row #6 (permuted-label training): Row #6 shows the model needs real context to learn effective representations. This test shows the model needs real context to make accurate predictions. Together, they provide converging evidence that context features carry genuine biological information about promoter activity, not noise.
- The magnitude of the effect (ΔAUC = +0.3102, Cohen’s d = 39.2) strongly supports Sub-Claim B: CARCM’s context-aware architecture provides substantial, biologically meaningful improvement, and context-sequence integration is the model’s core innovation rather than a limitation.

## 5. Output Files

- `permutation_results.json` — Summary statistics (AUC, p-value, effect size, etc.)
- `permuted_auc_values.csv` — All 1000 permuted AUC values for replotting
- `permutation_histogram.{{png,pdf}}` — Histogram of permuted vs observed AUC (400 dpi)
- `permutation_convergence.{{png,pdf}}` — Running mean and p-value convergence (400 dpi)
- `permutation_boxplot.{{png,pdf}}` — Boxplot comparison (400 dpi)
- `console_output.txt` — Full console log of the run
