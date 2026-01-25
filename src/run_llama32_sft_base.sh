#!/bin/bash
# ============================================================================
# SFT Base Training - Llama 3.2 1B
# ============================================================================

# --- Configuration ---
MODEL="llama3_1"            # Make sure this matches your config.yaml key
EXP_NAME="llama32_1b_sft_filtered_ep2"
N_EPOCHS=2
N_EXAMPLES=6464
DATASET_SIZE=3232
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
EVAL_EVERY=500
LR="5e-7"                     # Standard SFT LR

echo "=============================================="
echo "Starting SFT Base: ${EXP_NAME}"
echo "Model: ${MODEL} | GPU: 0"
echo "=============================================="

python -u train.py \
    model=${MODEL} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    train_split=filtered_dpo \
    dataset_size=$DATASET_SIZE \
    n_epochs=${N_EPOCHS} \
    n_examples=${N_EXAMPLES} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    eval_every=${EVAL_EVERY} \
    lr=${LR} \
    save_ckp=true
