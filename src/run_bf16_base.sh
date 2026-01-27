#!/bin/bash
# ============================================================================
# SFT Base Training - Llama 3.2 1B
# ============================================================================
set -e
export CUDA_VISIBLE_DEVICES=0
bash run_bf16_sft_base.sh
bash run_bf16_dpo_base.sh
EXP_NAME="llama32_1b_dpo_ep6_bf16"

python -u gen_multipt.py \
        model=llama3_1\
        model.archive=$EXP_NAME \
        exp_name="eval_$EXP_NAME"
bash score.sh
