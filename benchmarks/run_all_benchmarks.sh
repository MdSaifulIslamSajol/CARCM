#!/usr/bin/env bash
# =============================================================================
# Regenerate everything in benchmarks/results/summary/ from scratch:
#   1) run_simple_baselines.py   -> GaussianNB, BernoulliNB, KNN, DecisionTree
#   2) run_classical.py          -> LogisticRegression, SVM_RBF, RandomForest,
#                                   XGBoost, LightGBM, CatBoost
#   3) run_tabnet.py             -> TabNet
#   4) run_ft_transformer.py     -> FTTransformer
#   5) aggregate_and_plot.py     -> CSV, LaTeX table, plots in results/summary/
#
# Assumes V7b stacking outputs already exist at
#   /home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/models_v7b/lr_1e-3/
# (aggregate_and_plot.py reads lr_1e-3_results.json / lr_1e-3_predictions.csv).
#
# Logs: benchmarks/logs/<step>.log
# =============================================================================
set -euo pipefail

PYTHON="/home/saiful/anaconda3/envs/dnabert_bench/bin/python"
BENCH_DIR="/home/saiful/CARCM/benchmarks"
LOG_DIR="$BENCH_DIR/logs"
SUMMARY_DIR="$BENCH_DIR/results/summary"

mkdir -p "$LOG_DIR"
cd "$BENCH_DIR"

SECONDS=0

echo "========================================================"
echo "  Motif-Based Classification -- Full Benchmark Pipeline"
echo "  Python   : $PYTHON"
echo "  Bench dir: $BENCH_DIR"
echo "  Started  : $(date)"
echo "========================================================"

# ========================== 1/5  Simple baselines ==========================
echo ""
echo "##################  [1/5] run_simple_baselines.py  ##################"
T0=$SECONDS
$PYTHON run_simple_baselines.py --models all \
    2>&1 | tee "$LOG_DIR/run_simple_baselines.log"
echo ">>> run_simple_baselines DONE ($((SECONDS - T0))s)"

# ========================== 2/5  Classical ML ==============================
echo ""
echo "##################  [2/5] run_classical.py  ##################"
T0=$SECONDS
$PYTHON run_classical.py --models all \
    2>&1 | tee "$LOG_DIR/run_classical.log"
echo ">>> run_classical DONE ($((SECONDS - T0))s)"

# ========================== 3/5  TabNet ====================================
echo ""
echo "##################  [3/5] run_tabnet.py  ##################"
T0=$SECONDS
$PYTHON run_tabnet.py \
    2>&1 | tee "$LOG_DIR/run_tabnet.log"
echo ">>> run_tabnet DONE ($((SECONDS - T0))s)"

# ========================== 4/5  FT-Transformer ============================
echo ""
echo "##################  [4/5] run_ft_transformer.py  ##################"
T0=$SECONDS
$PYTHON run_ft_transformer.py \
    2>&1 | tee "$LOG_DIR/run_ft_transformer.log"
echo ">>> run_ft_transformer DONE ($((SECONDS - T0))s)"

# ========================== 5/5  Aggregate + Plot ==========================
echo ""
echo "##################  [5/5] aggregate_and_plot.py  ##################"
T0=$SECONDS
$PYTHON aggregate_and_plot.py \
    2>&1 | tee "$LOG_DIR/aggregate_and_plot.log"
echo ">>> aggregate_and_plot DONE ($((SECONDS - T0))s)"

TOTAL_TIME=$SECONDS
echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE"
echo "  Total time : $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
echo "  Finished   : $(date)"
echo "  Summary    : $SUMMARY_DIR"
echo "  Logs       : $LOG_DIR"
echo "========================================================"
