#!/bin/bash
# Context Permutation Test — "The Context Signal Exists" Test
# 1000 permutation iterations at inference time (no retraining)
set -e
cd "/home/saiful/CARCM/Permutation test -the context signal exists test"

echo "Running context permutation test (1000 iterations)..."
echo "This will take a while — each iteration runs a 10-model ensemble on the test set."
echo ""

conda run --no-capture-output -n dnabert \
    python context_permutation_test.py \
    2>&1 | tee console_output.txt

echo ""
echo "Done — results saved in: $(pwd)/"
