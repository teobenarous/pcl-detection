#!/bin/bash
# Usage: ./run_experiment.sh <output_dir> <model_name_or_path> <use_llrd> <use_focal> <seed>

set -e

# (1) ABLATION TOGGLES
OUTPUT_DIR=$1
MODEL_PATH=$2
USE_LLRD=$3
USE_FOCAL=$4
SEED=${5:-42}

echo "================================================================"
echo "Running Experiment: $OUTPUT_DIR"
echo "Model: $MODEL_PATH | LLRD: $USE_LLRD | Focal: $USE_FOCAL"
echo "================================================================"

# (2) PARAMETERS

# Data
TRAIN_FILE="data/processed/train.csv"
VAL_FILE="data/processed/dev_internal.csv"
MAX_SEQ_LEN=256

# Core
EPOCHS=10
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
GRAD_ACCUM_STEPS=4
BF16="True"

# Optimizer & Scheduler
LR=1e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1

# LLRD
LLRD_DECAY=0.95

# Focal Loss
FOCAL_GAMMA=1.0
FOCAL_ALPHA=0.75

# (3) RUN

python train.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$VAL_FILE" \
    --max_seq_length $MAX_SEQ_LEN \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --bf16 $BF16 \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --use_llrd "$USE_LLRD" \
    --llrd_decay $LLRD_DECAY \
    --use_focal_loss "$USE_FOCAL" \
    --focal_gamma $FOCAL_GAMMA \
    --focal_alpha $FOCAL_ALPHA \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "auprc" \
    --report_to "none"

echo "Finished training for $OUTPUT_DIR"
echo "-----------------------------------------------------------------"