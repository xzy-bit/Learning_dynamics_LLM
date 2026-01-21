#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "${SCRIPT_DIR}/../src"

# --- Configuration ---
MODEL="llama3_1"
SFT_CHECKPOINT="llama32_1b_sft_base_ep2"
EXP_NAME="llama32_1b_dpo_base_ep6"

N_EPOCHS=6
N_EXAMPLES=30000              # 5000 examples per epoch
DATASET_SIZE=5000
# === MEMORY SAFETY ===
# DPO loads 2 models. We keep Global Batch 32, but slice it smaller (Micro 8).
BATCH_SIZE=32                 
GRADIENT_ACCUMULATION_STEPS=4 

EVAL_EVERY=1000
LR="5e-7"
BETA=0.1
SAVE_EPOCHS="[1,2,3,4,5,6]"

if [ ! -f "exp_results/${SFT_CHECKPOINT}/policy.pt" ]; then
    echo "ERROR: SFT checkpoint not found at exp_results/${SFT_CHECKPOINT}/policy.pt"
    exit 1
fi

echo "=============================================="
echo "Starting DPO Base: ${EXP_NAME}"
echo "Source: ${SFT_CHECKPOINT}"
echo "=============================================="

python -u train.py \
    loss=dpo \
    loss.beta=${BETA} \
    model=${MODEL} \
    dataset_size=$DATASET_SIZE \
    model.archive=${SFT_CHECKPOINT} \
    exp_name=${EXP_NAME} \
    trainer=BasicTrainer \
    train_split=train_dpo \
    n_epochs=${N_EPOCHS} \
    n_examples=${N_EXAMPLES} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    eval_every=${EVAL_EVERY} \
    lr=${LR} \
    save_ckp=true