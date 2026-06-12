#!/bin/bash
# =============================================================================
# CARCM — 5-Fold Cross-Validation (GPUs 2-6 in parallel)
# =============================================================================
# Step 1: Feature extraction + 5-fold stratified split   (CPU only, ~15 min)
# Step 2: Train classifier on all 5 folds simultaneously (GPUs 2,3,4,5,6)
# Step 3: Aggregate results + comparison plots
#
# Best config (from V7b search):
#   LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10
#
# Usage:
#   chmod +x run_cv.sh
#   ./run_cv.sh
# =============================================================================

set -e

cd /home/saiful/CARCM/

CV_DIR="benchmarks/cross_validation"
CONSOLE_DIR="${CV_DIR}/console_output"
mkdir -p "$CONSOLE_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  CARCM — 5-FOLD CROSS-VALIDATION                            ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Step 1 │ Feature extraction + 5-fold stratified split  (CPU)               ║"
echo "║  Step 2 │ Train CARCM on folds 0-4  (GPUs 2,3,4,5,6 in parallel)           ║"
echo "║  Step 3 │ Aggregate results + comparison plots                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Feature Extraction + 5-Fold Split ──────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [Step 1/3] Feature Extraction + Stratified 5-Fold Splitting                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

conda run --no-capture-output -n dnabert \
    python ${CV_DIR}/cv_pipeline.py \
    2>&1 | tee "${CONSOLE_DIR}/cv_pipeline.txt"

echo ""
echo "  ✓ Pipeline complete — all 5 fold splits created."
echo ""

# ─── Step 2: Train All 5 Folds in Parallel ──────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [Step 2/3] Training CARCM on 5 folds (GPUs 2-6, parallel)                 ║"
echo "║  Config: LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

GPUS=(2 3 4 5 6)
PIDS=()

for fold in 0 1 2 3 4; do
    GPU=${GPUS[$fold]}
    echo "  Launching fold ${fold} on GPU ${GPU} ..."
    conda run --no-capture-output -n dnabert \
        python ${CV_DIR}/cv_classifier.py \
            --fold ${fold} \
            --gpu ${GPU} \
            --learning_rate 1e-3 \
            --hidden_dim 128 \
            --dropout 0.4 \
            --batch_size 128 \
            --num_seeds 10 \
            --num_epochs 150 \
            --patience 25 \
        > "${CONSOLE_DIR}/fold_${fold}_classifier.txt" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "  All 5 folds launched. Waiting for completion..."
echo "  (Monitor with: tail -f ${CONSOLE_DIR}/fold_*_classifier.txt)"
echo ""

# Wait for each fold and report status
FAILED=0
for i in 0 1 2 3 4; do
    if wait ${PIDS[$i]}; then
        echo "  ✓ Fold ${i} complete  (GPU ${GPUS[$i]}, PID ${PIDS[$i]})"
    else
        echo "  ✗ Fold ${i} FAILED   (GPU ${GPUS[$i]}, PID ${PIDS[$i]})"
        FAILED=1
    fi
done

echo ""
if [ $FAILED -eq 1 ]; then
    echo "  WARNING: One or more folds failed. Check console outputs in ${CONSOLE_DIR}/"
fi

# ─── Step 3: Aggregate Results ───────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [Step 3/3] Aggregating Cross-Validation Results                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

conda run --no-capture-output -n dnabert \
    python ${CV_DIR}/cv_aggregate.py \
    2>&1 | tee "${CONSOLE_DIR}/cv_aggregate.txt"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✓ DONE — 5-Fold Cross-Validation Complete                                 ║"
echo "║                                                                             ║"
echo "║  Results in: benchmarks/cross_validation/                                   ║"
echo "║    cv_per_fold_metrics.csv        — per-fold metrics                        ║"
echo "║    cv_summary.csv                 — mean ± std                              ║"
echo "║    cv_concatenated_predictions.csv— full-dataset predictions                ║"
echo "║    cv_full_dataset_metrics.json   — pooled CV metrics                       ║"
echo "║    cv_fold_comparison.{png,pdf}   — bar-chart comparison                    ║"
echo "║    cv_radar_chart.{png,pdf}       — radar overlay                           ║"
echo "║    console_output/                — all console logs                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
