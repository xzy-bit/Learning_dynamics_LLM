#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

MODEL="qwen18"
DATASET="hh"
SFT_EPOCHS=2
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=100000
DATE="1223"

MASK_TYPE="ratio"
MASK_TOP_K=50
MASK_STRENGTH=0.0
MASK_THRESHOLD_PROB=0.1

RATIOS=(0.99 0.999 0.9999 0.99999 0.999999)

for MASK_RATIO in "${RATIOS[@]}"; do
    echo "======================================================"
    echo "Running Mask-DPO with mask_ratio=${MASK_RATIO}"
    echo "======================================================"

    EXP_NAME="dpo_mask_${MASK_TYPE}_r_${MASK_RATIO}_k_${MASK_TOP_K}_s_${MASK_STRENGTH}_t_${MASK_THRESHOLD_PROB}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"

    # ------------------ Train ------------------
    python -u train.py \
        loss=masked_dpo \
        loss.beta=0.1 \
        loss.mask_type=${MASK_TYPE} \
        loss.mask_ratio=${MASK_RATIO} \
        loss.mask_top_k=${MASK_TOP_K} \
        loss.mask_strength=${MASK_STRENGTH} \
        loss.mask_threshold_prob=${MASK_THRESHOLD_PROB} \
        datasets=${DATASET} \
        model=${MODEL} \
        exp_name=${EXP_NAME} \
        trainer=BasicTrainer \
        n_epochs=${N_EPOCHS} \
        n_examples=${N_EXAMPLES} \
        model.archive="sft_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_1202" \
        save_ckp=true \
        eval_every=${EVAL_EVERY}

    # ------------------ Generate / Eval ------------------
    python -u gen_multipt.py \
        model=${MODEL} \
        model.archive=${EXP_NAME} \
        exp_name="eval_${EXP_NAME}"

done
bash score.sh
