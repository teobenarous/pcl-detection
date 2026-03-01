#!/bin/bash

set -e

BASE_MODEL="models/roberta-large"
ABLATION_DIR="experiments/ablation"

echo "Starting Ablation Study..."
mkdir -p "$ABLATION_DIR"

echo "========================================"
echo "Experiment 1: Baseline Only"
./run_experiment.sh "$ABLATION_DIR/1_roberta" "$BASE_MODEL" "False" "False"

echo "========================================"
echo "Experiment 2: Baseline + LLRD"
./run_experiment.sh "$ABLATION_DIR/2_roberta_llrd" "$BASE_MODEL" "True" "False"

echo "========================================"
echo "Experiment 3: Baseline + Focal Loss"
./run_experiment.sh "$ABLATION_DIR/3_roberta_focal" "$BASE_MODEL" "False" "True"

echo "========================================"
echo "Experiment 4: Baseline + LLRD + Focal Loss"
./run_experiment.sh "$ABLATION_DIR/4_roberta_llrd_focal" "$BASE_MODEL" "True" "True"

echo "All ablation experiments successfully completed!"