#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3

#bash run_llama32_sft_extend.sh
#bash run_llama32_dpo_extend.sh
EXP_NAME="llama32_1b_dpo_extend_filtered_ep6"

python -u gen_multipt.py \
        model=llama3_1\
        model.archive=$EXP_NAME \
        exp_name="eval_$EXP_NAME"
bash score.sh
