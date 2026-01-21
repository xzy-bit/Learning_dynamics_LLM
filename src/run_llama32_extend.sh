#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

bash run_llama32_sft_extend.sh
bash run_llama32_dpo_extend.sh