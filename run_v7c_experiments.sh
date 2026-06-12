#!/bin/bash
# =============================================================================
# Rice Promoter Classifier V7c - Best Configuration from V7b Search
# =============================================================================
# Best config: lr_1e-3 (LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10)
# Source: models_v7b/comprehensive_results.csv — highest Composite=0.7823, AUC=0.8423
# =============================================================================
# Usage:
#   chmod +x run_v7c_experiments.sh
#   ./run_v7c_experiments.sh
# =============================================================================

cd /home/saiful/CARCM/

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          V7c — BEST CONFIGURATION FROM V7b HYPERPARAMETER SEARCH             ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Run Name  │ LR      │ Hidden │ Dropout │ Batch │ Seeds                      ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  best      │ 1e-3    │ 128    │ 0.4     │ 128   │ 10                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting experiment..."
echo ""

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [1/1] RUNNING: best                                                         ║"
echo "║  Hyperparameters: LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
conda run -n dnabert python rice_classifier_v7b.py --learning_rate 1e-3 --hidden_dim 128 --dropout 0.4 --batch_size 128 --num_seeds 10 --run_name best

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  DONE — V7c experiment complete                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
