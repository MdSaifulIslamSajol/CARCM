#!/bin/bash
# =============================================================================
# CARCM-SALT — V7b Hyperparameter Search (mirrors drought run_v7b_experiments.sh)
# =============================================================================
# 15 configs: one-factor-at-a-time around baseline (3e-4/128/0.4/128/10seeds)
# + 3 combined configs. All run on the single 80/10/10 split (like drought).
# Distributed across 8 A100s (gpu = job_index % 8); the salt model is tiny so
# two jobs per GPU is fine. Results -> models_carcm_salt/<run_name>/.
#
# Assumes features already exist (output/). If not, run the pipeline first:
#   conda run -n dnabert python rice_positional_pipeline_salt.py
# =============================================================================
set -e
cd /home/saiful/DeepRice_Salt/CARCM_Salt/
mkdir -p hpsearch_logs

# run_name  lr     hidden dropout batch seeds
CONFIGS=(
  "baseline      3e-4 128 0.4 128 10"
  "lr_1e-4       1e-4 128 0.4 128 10"
  "lr_5e-4       5e-4 128 0.4 128 10"
  "lr_1e-3       1e-3 128 0.4 128 10"
  "hd_64         3e-4 64  0.4 128 10"
  "hd_256        3e-4 256 0.4 128 10"
  "do_0.3        3e-4 128 0.3 128 10"
  "do_0.5        3e-4 128 0.5 128 10"
  "bs_64         3e-4 128 0.4 64  10"
  "bs_256        3e-4 128 0.4 256 10"
  "ns_5          3e-4 128 0.4 128 5"
  "ns_15         3e-4 128 0.4 128 15"
  "combined_opt1 1e-4 128 0.5 128 10"
  "combined_opt2 3e-4 256 0.5 64  10"
  "combined_opt3 5e-4 64  0.3 256 10"
)

i=0
PIDS=()
for cfg in "${CONFIGS[@]}"; do
  read -r name lr hd do bs ns <<< "$cfg"
  gpu=$(( i % 8 ))
  echo "[$i] $name -> GPU $gpu  (lr=$lr hd=$hd do=$do bs=$bs seeds=$ns)"
  CARCM_SALT_GPU=$gpu python rice_classifier_v7b_salt.py \
      --learning_rate $lr --hidden_dim $hd --dropout $do \
      --batch_size $bs --num_seeds $ns --run_name "$name" \
      > "hpsearch_logs/${name}.log" 2>&1 &
  PIDS+=($!)
  i=$(( i + 1 ))
done

echo ""
echo "Launched ${#PIDS[@]} jobs. Waiting..."
FAIL=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then echo "  ok: job $idx"; else echo "  FAIL: job $idx"; FAIL=1; fi
done
echo ""
[ $FAIL -eq 0 ] && echo "ALL DONE" || echo "SOME FAILED — check hpsearch_logs/"
