#!/bin/bash
# =============================================================================
# V7b hyperparameter search -- DEV-SELECTED (corrected protocol)
# =============================================================================
# Identical 15 configurations to run_v7b_experiments.sh. The only difference is
# that rice_classifier_v7b.py now selects the ensemble aggregation strategy and
# tunes the decision threshold on the development split, and
# aggregate_v7b_results.py ranks configurations by dev AUC.
#
# Runs 8 configurations concurrently, one per GPU (GPU g takes configs g, g+8).
# Logs: logs_dev_search/<run_name>.log
# =============================================================================

cd /home/saiful/CARCM/
mkdir -p logs_dev_search

CONFIGS=(
  "3e-4 128 0.4 128 10 baseline"
  "1e-4 128 0.4 128 10 lr_1e-4"
  "5e-4 128 0.4 128 10 lr_5e-4"
  "1e-3 128 0.4 128 10 lr_1e-3"
  "3e-4 64  0.4 128 10 hd_64"
  "3e-4 256 0.4 128 10 hd_256"
  "3e-4 128 0.3 128 10 do_0.3"
  "3e-4 128 0.5 128 10 do_0.5"
  "3e-4 128 0.4 64  10 bs_64"
  "3e-4 128 0.4 256 10 bs_256"
  "3e-4 128 0.4 128 5  ns_5"
  "3e-4 128 0.4 128 15 ns_15"
  "1e-4 128 0.5 128 10 combined_opt1"
  "3e-4 256 0.5 64  10 combined_opt2"
  "5e-4 64  0.3 256 10 combined_opt3"
)

run_one() {
  read -r LR HD DO BS NS NAME <<< "$1"
  local GPU=$2
  echo "[GPU $GPU] START $NAME (lr=$LR hd=$HD do=$DO bs=$BS ns=$NS) $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n dnabert python rice_classifier_v7b.py \
      --learning_rate "$LR" --hidden_dim "$HD" --dropout "$DO" \
      --batch_size "$BS" --num_seeds "$NS" --run_name "$NAME" \
      > "logs_dev_search/${NAME}.log" 2>&1
  echo "[GPU $GPU] DONE  $NAME (exit $?) $(date +%H:%M:%S)"
}

# GPU g processes configs g, g+8 sequentially; the 8 GPU lanes run in parallel.
for GPU in 0 1 2 3 4 5 6 7; do
  (
    for ((i = GPU; i < ${#CONFIGS[@]}; i += 8)); do
      run_one "${CONFIGS[$i]}" "$GPU"
    done
  ) &
done

wait
echo ""
echo "=== ALL 15 CONFIGURATIONS COMPLETE $(date) ==="
