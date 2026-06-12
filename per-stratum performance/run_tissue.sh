#!/bin/bash
# Table S-Tissue: per-tissue AUC analysis
set -e
cd "/home/saiful/CARCM/per-stratum performance"
echo "Running per-tissue stratum analysis..."
conda run --no-capture-output -n dnabert python per_stratum_tissue.py 2>&1 | tee tissue/console_output.txt
echo "Done — results in tissue/"
