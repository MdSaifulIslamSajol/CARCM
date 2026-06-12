#!/bin/bash
# =============================================================================
# CARCM LOI (Leave-One-In) Feature Ablation — master runner
# =============================================================================
# Trains four configs (only_binned, only_positional, only_count, only_cooc),
# each with 10 seeds × 150 epochs × patience 25 (matching v7c regime), one
# config per GPU on GPUs 2, 3, 4, 6, then aggregates into:
#   results/loi_aggregate_metrics.csv
#   figures/loi_delta_auc.{png,pdf}
#   report/loi_report.{md,docx}
#
# Usage:
#   chmod +x run_loi.sh
#   ./run_loi.sh
# =============================================================================
set -u

cd "$(dirname "$0")"

LOI_ROOT="$(pwd)"
LOG_DIR="${LOI_ROOT}/console_outputs"
RESULTS_DIR="${LOI_ROOT}/results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

MASTER_LOG="${LOG_DIR}/master.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

NUM_SEEDS="${NUM_SEEDS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-150}"
PATIENCE="${PATIENCE:-25}"
CONDA_ENV="dnabert"

if [ -f /home/saiful/anaconda3/etc/profile.d/conda.sh ]; then
  source /home/saiful/anaconda3/etc/profile.d/conda.sh
elif [ -f /home/saiful/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/saiful/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${CONDA_ENV}"

echo "=============================================================================="
echo " CARCM LOI FEATURE ABLATION"
echo "=============================================================================="
echo " conda env       : ${CONDA_ENV}"
echo " num_seeds       : ${NUM_SEEDS}"
echo " num_epochs      : ${NUM_EPOCHS}    patience : ${PATIENCE}"
echo " loi root        : ${LOI_ROOT}"
echo " log dir         : ${LOG_DIR}"
echo " results dir     : ${RESULTS_DIR}"
echo " GPU schedule    : only_binned=2, only_positional=3, only_count=4, only_cooc=6"
echo "=============================================================================="
echo " start: $(date)"
echo

LAUNCHED_PIDS=()
run_cfg () {
  local cfg_name="$1"
  local gpu="$2"
  local log="${LOG_DIR}/${cfg_name}.log"
  echo "[launch] cfg=${cfg_name}  gpu=${gpu}  log=${log}"
  python -u loi_train.py \
      --config "${cfg_name}" \
      --gpu "${gpu}" \
      --num_seeds "${NUM_SEEDS}" \
      --num_epochs "${NUM_EPOCHS}" \
      --patience "${PATIENCE}" \
      > "${log}" 2>&1 &
  LAUNCHED_PIDS+=("$!")
}

echo "============================ WAVE 1 (LOI, 4 configs) ============================"
LAUNCHED_PIDS=()
run_cfg only_binned     2
run_cfg only_positional 3
run_cfg only_count      4
run_cfg only_cooc       6
echo "wave1 PIDs: ${LAUNCHED_PIDS[*]}"
for pid in "${LAUNCHED_PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "[ERROR] LOI PID ${pid} exited non-zero"
  fi
done
echo "wave 1 complete: $(date)"

echo
echo "============================ AGGREGATE ============================"
python -u loi_aggregate.py --root "${LOI_ROOT}/" \
    > "${LOG_DIR}/aggregate.log" 2>&1
echo "aggregate complete: $(date)"

echo
echo "=============================================================================="
echo " LOI ABLATION DONE  -- $(date)"
echo "  CSV     : ${RESULTS_DIR}/loi_aggregate_metrics.csv"
echo "  CSV     : ${RESULTS_DIR}/loi_per_seed_long.csv"
echo "  CSV     : ${RESULTS_DIR}/loi_delong.csv"
echo "  Figure  : ${LOI_ROOT}/figures/loi_delta_auc.{png,pdf}"
echo "  Figure  : ${LOI_ROOT}/figures/loi_metric_grid.{png,pdf}"
echo "  Report  : ${LOI_ROOT}/report/loi_report.{md,docx}"
echo "=============================================================================="
