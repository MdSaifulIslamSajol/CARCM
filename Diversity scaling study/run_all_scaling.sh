#!/usr/bin/env bash
# ==============================================================================
# run_all_scaling.sh
# ------------------------------------------------------------------------------
# Launches the three panels (variety, tissue, stage) of the diversity-scaling
# study in parallel on separate GPUs, then aggregates all results and writes
# the .docx + .md report.
#
# GPUs used (chosen at launch time based on available memory):
#   variety -> GPU 2
#   tissue  -> GPU 4
#   stage   -> GPU 7
# ==============================================================================
set -u -o pipefail

WORK_DIR="/home/saiful/CARCM/Diversity scaling study"
CONSOLE_DIR="${WORK_DIR}/console_output"
mkdir -p "${CONSOLE_DIR}"

export PYTHONUNBUFFERED=1
# Ensure CUDA device indices match nvidia-smi indices (by default CUDA uses
# FASTEST_FIRST ordering which can re-map GPUs and cause all subshells to
# land on the same physical device even with distinct CUDA_VISIBLE_DEVICES).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=============================================================="
echo "Diversity scaling study - $(date)"
echo "Working dir: ${WORK_DIR}"
echo "=============================================================="

# ----- Launch each panel on its own GPU in the background ---------------------
echo "[launch] variety -> GPU 2"
(
  cd "${WORK_DIR}"
  CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n dnabert \
    python -u run_scaling.py --panel variety \
    > "${CONSOLE_DIR}/variety_scaling.txt" 2>&1
) &
PID_VAR=$!

echo "[launch] tissue  -> GPU 4"
(
  cd "${WORK_DIR}"
  CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n dnabert \
    python -u run_scaling.py --panel tissue \
    > "${CONSOLE_DIR}/tissue_scaling.txt" 2>&1
) &
PID_TIS=$!

echo "[launch] stage   -> GPU 7"
(
  cd "${WORK_DIR}"
  CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n dnabert \
    python -u run_scaling.py --panel stage \
    > "${CONSOLE_DIR}/stage_scaling.txt" 2>&1
) &
PID_STG=$!

echo "[launch] PIDs: variety=${PID_VAR}  tissue=${PID_TIS}  stage=${PID_STG}"
echo "[launch] tail the logs at:"
echo "         ${CONSOLE_DIR}/variety_scaling.txt"
echo "         ${CONSOLE_DIR}/tissue_scaling.txt"
echo "         ${CONSOLE_DIR}/stage_scaling.txt"

# ----- Wait for all three panels to finish -----------------------------------
wait ${PID_VAR}; RC_VAR=$?
wait ${PID_TIS}; RC_TIS=$?
wait ${PID_STG}; RC_STG=$?

echo "=============================================================="
echo "[panels done] variety rc=${RC_VAR}  tissue rc=${RC_TIS}  stage rc=${RC_STG}"

# ----- Aggregate & report ----------------------------------------------------
echo "[aggregate] aggregating CSVs + figures ..."
cd "${WORK_DIR}"
conda run --no-capture-output -n dnabert python -u aggregate_scaling.py \
  > "${CONSOLE_DIR}/aggregate.txt" 2>&1
echo "[aggregate] done. See ${CONSOLE_DIR}/aggregate.txt"

echo "[report] writing .docx + .md ..."
conda run --no-capture-output -n dnabert python -u save_report.py \
  > "${CONSOLE_DIR}/report.txt" 2>&1
echo "[report] done. See ${CONSOLE_DIR}/report.txt"

echo "=============================================================="
echo "ALL DONE - $(date)"
echo "Outputs: ${WORK_DIR}/diversity_scaling_report.docx"
echo "         ${WORK_DIR}/diversity_scaling_report.md"
echo "=============================================================="
