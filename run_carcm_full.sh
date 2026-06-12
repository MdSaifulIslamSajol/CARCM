#!/bin/bash
# =============================================================================
# CARCM Full Pipeline — Feature Extraction + Classifier Training
# =============================================================================
# Step 1: rice_positional_pipeline.py — extracts positional motif features
# Step 2: rice_classifier_v7b.py      — trains the best CARCM configuration
#
# Best config (from V7b hyperparameter search):
#   LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10
#
# Usage:
#   chmod +x run_carcm_full.sh
#   ./run_carcm_full.sh
# =============================================================================

set -e  # exit on first error

cd /home/saiful/CARCM/

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    CARCM — FULL PIPELINE                                    ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Step 1 │ Positional motif feature extraction                               ║"
echo "║  Step 2 │ Classifier training (best config: LR=1e-3, H=128, D=0.4, B=128)  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ─── Step 1: Feature Extraction ──────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [1/2] RUNNING: Positional Motif Feature Extraction                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

conda run -n dnabert python rice_positional_pipeline.py

echo ""
echo "  ✓ Feature extraction complete."
echo ""

# ─── Step 2: Classifier Training ─────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [2/2] RUNNING: CARCM Classifier (best configuration)                      ║"
echo "║  Hyperparameters: LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

conda run -n dnabert python rice_classifier_v7b.py \
    --learning_rate 1e-3 \
    --hidden_dim 128 \
    --dropout 0.4 \
    --batch_size 128 \
    --num_seeds 10 \
    --run_name best

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✓ DONE — CARCM full pipeline complete                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
