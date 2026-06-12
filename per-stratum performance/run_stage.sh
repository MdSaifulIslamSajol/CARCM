#!/bin/bash
# Table S-Stage: per-stage AUC analysis
set -e
cd "/home/saiful/CARCM/per-stratum performance"
echo "Running per-stage stratum analysis..."
conda run --no-capture-output -n dnabert python per_stratum_stage.py 2>&1 | tee stage/console_output.txt
echo "Done — results in stage/"
