#!/bin/bash
# =============================================================================
# Rice Promoter Classifier V7b - Hyperparameter Search Script
# =============================================================================
# Usage:
#   chmod +x run_v7b_experiments.sh
#   ./run_v7b_experiments.sh
# =============================================================================

cd /home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/

# =============================================================================
# PRINT ALL HYPERPARAMETER COMBINATIONS FIRST
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          V7b HYPERPARAMETER SEARCH - 15 CONFIGURATIONS                       ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  #   │ Run Name       │ LR      │ Hidden │ Dropout │ Batch │ Seeds          ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║   1  │ baseline       │ 3e-4    │ 128    │ 0.4     │ 128   │ 10             ║"
echo "║   2  │ lr_1e-4        │ 1e-4    │ 128    │ 0.4     │ 128   │ 10             ║"
echo "║   3  │ lr_5e-4        │ 5e-4    │ 128    │ 0.4     │ 128   │ 10             ║"
echo "║   4  │ lr_1e-3        │ 1e-3    │ 128    │ 0.4     │ 128   │ 10             ║"
echo "║   5  │ hd_64          │ 3e-4    │ 64     │ 0.4     │ 128   │ 10             ║"
echo "║   6  │ hd_256         │ 3e-4    │ 256    │ 0.4     │ 128   │ 10             ║"
echo "║   7  │ do_0.3         │ 3e-4    │ 128    │ 0.3     │ 128   │ 10             ║"
echo "║   8  │ do_0.5         │ 3e-4    │ 128    │ 0.5     │ 128   │ 10             ║"
echo "║   9  │ bs_64          │ 3e-4    │ 128    │ 0.4     │ 64    │ 10             ║"
echo "║  10  │ bs_256         │ 3e-4    │ 128    │ 0.4     │ 256   │ 10             ║"
echo "║  11  │ ns_5           │ 3e-4    │ 128    │ 0.4     │ 128   │ 5              ║"
echo "║  12  │ ns_15          │ 3e-4    │ 128    │ 0.4     │ 128   │ 15             ║"
echo "║  13  │ combined_opt1  │ 1e-4    │ 128    │ 0.5     │ 128   │ 10             ║"
echo "║  14  │ combined_opt2  │ 3e-4    │ 256    │ 0.5     │ 64    │ 10             ║"
echo "║  15  │ combined_opt3  │ 5e-4    │ 64     │ 0.3     │ 256   │ 10             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Estimated total time: 4-6 hours"
echo "Starting experiments..."
echo ""

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [1/15] RUNNING: baseline                                                    ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=128, Dropout=0.4, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 128 --dropout 0.4 --batch_size 128 --num_seeds 10 --run_name baseline

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [2/15] RUNNING: lr_1e-4                                                     ║"
echo "║  Hyperparameters: LR=1e-4, Hidden=128, Dropout=0.4, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 1e-4 --hidden_dim 128 --dropout 0.4 --batch_size 128 --num_seeds 10 --run_name lr_1e-4

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [3/15] RUNNING: lr_5e-4                                                     ║"
echo "║  Hyperparameters: LR=5e-4, Hidden=128, Dropout=0.4, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 5e-4 --hidden_dim 128 --dropout 0.4 --batch_size 128 --num_seeds 10 --run_name lr_5e-4

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [4/15] RUNNING: lr_1e-3                                                     ║"
echo "║  Hyperparameters: LR=1e-3, Hidden=128, Dropout=0.4, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 1e-3 --hidden_dim 128 --dropout 0.4 --batch_size 128 --num_seeds 10 --run_name lr_1e-3

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [5/15] RUNNING: hd_64                                                       ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=64, Dropout=0.4, Batch=128, Seeds=10       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 64 --dropout 0.4 --batch_size 128 --num_seeds 10 --run_name hd_64

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [6/15] RUNNING: hd_256                                                      ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=256, Dropout=0.4, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 256 --dropout 0.4 --batch_size 128 --num_seeds 10 --run_name hd_256

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [7/15] RUNNING: do_0.3                                                      ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=128, Dropout=0.3, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 128 --dropout 0.3 --batch_size 128 --num_seeds 10 --run_name do_0.3

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [8/15] RUNNING: do_0.5                                                      ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=128, Dropout=0.5, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 128 --dropout 0.5 --batch_size 128 --num_seeds 10 --run_name do_0.5

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [9/15] RUNNING: bs_64                                                       ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=128, Dropout=0.4, Batch=64, Seeds=10       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 128 --dropout 0.4 --batch_size 64 --num_seeds 10 --run_name bs_64

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [10/15] RUNNING: bs_256                                                     ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=128, Dropout=0.4, Batch=256, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 128 --dropout 0.4 --batch_size 256 --num_seeds 10 --run_name bs_256

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [11/15] RUNNING: ns_5                                                       ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=128, Dropout=0.4, Batch=128, Seeds=5       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 128 --dropout 0.4 --batch_size 128 --num_seeds 5 --run_name ns_5

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [12/15] RUNNING: ns_15                                                      ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=128, Dropout=0.4, Batch=128, Seeds=15      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 128 --dropout 0.4 --batch_size 128 --num_seeds 15 --run_name ns_15

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [13/15] RUNNING: combined_opt1                                              ║"
echo "║  Hyperparameters: LR=1e-4, Hidden=128, Dropout=0.5, Batch=128, Seeds=10      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 1e-4 --hidden_dim 128 --dropout 0.5 --batch_size 128 --num_seeds 10 --run_name combined_opt1

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [14/15] RUNNING: combined_opt2                                              ║"
echo "║  Hyperparameters: LR=3e-4, Hidden=256, Dropout=0.5, Batch=64, Seeds=10       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 3e-4 --hidden_dim 256 --dropout 0.5 --batch_size 64 --num_seeds 10 --run_name combined_opt2

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  [15/15] RUNNING: combined_opt3                                              ║"
echo "║  Hyperparameters: LR=5e-4, Hidden=64, Dropout=0.3, Batch=256, Seeds=10       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
python rice_classifier_v7b.py --learning_rate 5e-4 --hidden_dim 64 --dropout 0.3 --batch_size 256 --num_seeds 10 --run_name combined_opt3

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ALL 15 EXPERIMENTS COMPLETED!                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results: /home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/models_v7b/"
echo "Checkpoints: /media/exxact1/saiful/DeepRice/Checkpoints/binbased_v2/"
echo ""
echo "Generating summary..."
python check_best_hyperparameters.py
echo ""
echo "Generating plots..."
python aggregate_v7b_results.py
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                              ALL DONE!                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
