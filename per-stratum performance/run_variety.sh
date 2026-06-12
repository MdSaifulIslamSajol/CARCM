#!/bin/bash
# Table S-Variety: per-variety AUC analysis
set -e
cd "/home/saiful/CARCM/per-stratum performance"
echo "Running per-variety stratum analysis..."
conda run --no-capture-output -n dnabert python per_stratum_variety.py 2>&1 | tee variety/console_output.txt
echo "Done — results in variety/"
