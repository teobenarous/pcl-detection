#!/bin/bash

set -e

BASE_SEED=${1:-42}
NUM_RUNS=5
MODEL_PATH="models/roberta-large"
ENSEMBLE_DIR="experiments/ensemble_llrd_focal"

echo "Starting Ensemble Training..."
echo "Base Seed: $BASE_SEED | Total Runs: $NUM_RUNS | Model: $MODEL_PATH"
mkdir -p "$ENSEMBLE_DIR"

for ((i=0; i<NUM_RUNS; i++)); do
    CURRENT_SEED=$((BASE_SEED + i))
    RUN_NUMBER=$((i + 1))
    OUTPUT_DIR="$ENSEMBLE_DIR/run${RUN_NUMBER}"

    mkdir -p "$OUTPUT_DIR"

    echo "================================================================="
    echo "Training Ensemble Model ${RUN_NUMBER} / $NUM_RUNS (SEED: $CURRENT_SEED)"
    echo "================================================================="

    # Arguments: <output_dir> <model_path> <use_llrd> <use_focal> <seed>
    ./run_experiment.sh "$OUTPUT_DIR" "$MODEL_PATH" "True" "True" "$CURRENT_SEED"

    echo "Model ${RUN_NUMBER} complete. Artifacts saved to: $OUTPUT_DIR"
    echo ""
done

echo "All $NUM_RUNS ensemble models have successfully finished training"