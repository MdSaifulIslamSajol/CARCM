#!/bin/bash
# =============================================================================
# Drought → Salt Transfer Learning — full run
# =============================================================================
# Step 1: build shared drought+salt feature space   (CPU, ~12 min)
# Step 2: pretrain on drought, fine-tune on salt, compare vs scratch (GPU)
#
# Tests "does shared abiotic-stress biology transfer?" by reusing the 14x
# larger drought corpus to initialise the CARCM encoder for salt.
# =============================================================================
set -e
cd /home/saiful/DeepRice_Salt/CARCM_Salt/transfer_learning/
GPU="${1:-0}"

echo "############ STEP 1: shared feature space ############"
conda run -n dnabert --no-capture-output python tl_features.py

echo ""
echo "############ STEP 2: transfer learning (GPU $GPU) ############"
# Hyperparameters are the ones selected on the salt development set
# (hidden 256, dropout 0.4, lr 3e-4) and reported in the manuscript.
conda run -n dnabert --no-capture-output python tl_transfer.py --gpu "$GPU" --num_seeds 5 \
    --hidden_dim 256 --dropout 0.4 --scratch_lr 3e-4 --finetune_lr 3e-4 --batch_size 128

echo ""
echo "DONE — results in transfer_learning/results/"
