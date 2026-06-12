#!/bin/bash
# =============================================================================
# Homology-Aware Gene-Level Split — Full Pipeline
# =============================================================================
# Steps:
#   1. Match RAP IDs + CD-HIT-EST clustering
#   2. Stratified cluster-level train/dev/test split
#   3. Feature extraction on the new split
#   4. Train CARCM classifier (best config)
#   5. Context ablation ladder
#   6. Comparison & report generation
# =============================================================================

set -e

WORK_DIR="/home/saiful/CARCM/Homology aware gene level split"
CONSOLE_DIR="${WORK_DIR}/console_output"
mkdir -p "${CONSOLE_DIR}"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          HOMOLOGY-AWARE GENE-LEVEL SPLIT — FULL PIPELINE                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Match RAP IDs + Cluster ──────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  [1/6] Matching RAP IDs + CD-HIT-EST clustering..."
echo "═══════════════════════════════════════════════════════════════════════"
conda run --no-capture-output -n dnabert python -u "${WORK_DIR}/step1_match_rapids_and_cluster.py" \
    2>&1 | tee "${CONSOLE_DIR}/step1_cluster.txt"
echo "  ✓ Step 1 complete."
echo ""

# ── Step 2: Stratified cluster split ─────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  [2/6] Creating stratified cluster-level split..."
echo "═══════════════════════════════════════════════════════════════════════"
conda run --no-capture-output -n dnabert python -u "${WORK_DIR}/step2_cluster_split.py" \
    2>&1 | tee "${CONSOLE_DIR}/step2_split.txt"
echo "  ✓ Step 2 complete."
echo ""

# ── Step 3: Feature extraction ───────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  [3/6] Extracting features with homology-aware split..."
echo "═══════════════════════════════════════════════════════════════════════"
conda run --no-capture-output -n dnabert python -u "${WORK_DIR}/step3_feature_extraction.py" \
    2>&1 | tee "${CONSOLE_DIR}/step3_features.txt"
echo "  ✓ Step 3 complete."
echo ""

# ── Step 4: Train CARCM classifier ──────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  [4/6] Training CARCM on homology-aware split (best config)..."
echo "═══════════════════════════════════════════════════════════════════════"
conda run --no-capture-output -n dnabert python -u "${WORK_DIR}/step4_train_classifier.py" \
    --learning_rate 1e-3 \
    --hidden_dim 128 \
    --dropout 0.4 \
    --batch_size 128 \
    --num_seeds 10 \
    --run_name best \
    2>&1 | tee "${CONSOLE_DIR}/step4_classifier.txt"
echo "  ✓ Step 4 complete."
echo ""

# ── Step 5: Context ablation ladder ──────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  [5/6] Running context ablation ladder..."
echo "═══════════════════════════════════════════════════════════════════════"
conda run --no-capture-output -n dnabert python -u "${WORK_DIR}/step5_context_ablation.py" \
    2>&1 | tee "${CONSOLE_DIR}/step5_ablation.txt"
echo "  ✓ Step 5 complete."
echo ""

# ── Step 6: Comparison & reports ─────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  [6/6] Generating comparison figures and reports..."
echo "═══════════════════════════════════════════════════════════════════════"
conda run --no-capture-output -n dnabert python -u "${WORK_DIR}/step6_compare_and_report.py" \
    2>&1 | tee "${CONSOLE_DIR}/step6_report.txt"
echo "  ✓ Step 6 complete."
echo ""

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✓ DONE — Homology-aware gene-level split pipeline complete                 ║"
echo "║                                                                              ║"
echo "║  Results in: ${WORK_DIR}                                                     ║"
echo "║  Reports:    homology_split_report.docx / .md                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
