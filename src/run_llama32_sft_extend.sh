#!/bin/bash
# ============================================================================
# SFT Extend Training - Llama 3.2 1B
# ============================================================================
set -e
export CUDA_VISIBLE_DEVICES=1

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "${SCRIPT_DIR}/../src"

# --- Configuration ---
MODEL="llama3_1"
EXP_NAME="llama32_1b_sft_extend_ep2"
N_EPOCHS=2
N_EXAMPLES=10000
DATASET_SIZE=50000
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2
EVAL_EVERY=500
LR="5e-7"

echo "=============================================="
echo "Starting SFT Extend: ${EXP_NAME}d"
echo "Model: ${MODEL} | GPU: 1"
echo "=============================================="

python -u train.py \
    model=${MODEL} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    train_split=train_sft_extend \
    dataset_size=$DATASET_SIZE \
    n_epochs=${N_EPOCHS} \
    n_examples=${N_EXAMPLES} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    eval_every=${EVAL_EVERY} \
    lr=${LR} \
    save_ckp=true