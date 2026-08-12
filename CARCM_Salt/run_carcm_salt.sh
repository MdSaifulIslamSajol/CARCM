#!/bin/bash
# =============================================================================
# CARCM on the SALT dataset — full pipeline
# =============================================================================
# Mirrors CARCM's drought "best" run:
#   LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10
# Steps:
#   1. Positional motif feature extraction (salinity motifs)  -> output/
#   2. Classifier train + ensemble + figures                  -> models_carcm_salt/best/
# =============================================================================
set -e
cd /home/saiful/DeepRice_Salt/CARCM_Salt/

GPU="${CARCM_SALT_GPU:-0}"
export CARCM_SALT_GPU="$GPU"

echo "=============================================================="
echo " CARCM-SALT  |  GPU=$GPU"
echo "=============================================================="

echo ""
echo "[1/2] Positional feature extraction (salinity motifs)..."
conda run -n dnabert --no-capture-output python rice_positional_pipeline_salt.py

echo ""
echo "[2/2] Classifier (best config: LR=1e-3 H=128 do=0.4 bs=128 seeds=10)..."
conda run -n dnabert --no-capture-output python rice_classifier_v7b_salt.py \
    --learning_rate 1e-3 --hidden_dim 128 --dropout 0.4 \
    --batch_size 128 --num_seeds 10 --run_name best

echo ""
echo "=============================================================="
echo " DONE — results in models_carcm_salt/best/"
echo "=============================================================="
