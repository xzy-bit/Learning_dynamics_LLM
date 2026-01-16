#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=3

BASE_DIR="./exp_results"
INPUT_NAME="prob_test_gen_response.jsonl"
OUTPUT_NAME="prmpt_resp_reward_new.jsonl"
RW_MODEL="Skywork/Skywork-Reward-V2-Llama-3.1-8B"

find "$BASE_DIR" -type f -name "$INPUT_NAME" > input_list.txt

python score_resp_batch.py \
    --input_list input_list.txt \
    --output_name "$OUTPUT_NAME" \
    --rw_model "$RW_MODEL" \
    --max_length 1024

