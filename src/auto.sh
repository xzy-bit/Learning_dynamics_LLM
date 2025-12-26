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
DATE="1226"

MASK_TYPE="ratio_log(1-p)"
MASK_TOP_K=0
MASK_THRESHOLD_PROB=0.1
MASK_RATIO = 0.1

MASK_STRENGTHS=(0.0 1.0 10.0 100.0 1000.0 10000.0 100000.0)

for MASK_STRENGTH in "${MASK_STRENGTHS[@]}"; do
    echo "======================================================"
    echo "Running Mask-DPO with mask_ratio=${MASK_STRENGTH}"
    echo "======================================================"

    EXP_NAME="dpo_mask_${MASK_TYPE}_r_${MASK_RATIO}_k_${MASK_TOP_K}_s_${MASK_STRENGTH}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"

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
