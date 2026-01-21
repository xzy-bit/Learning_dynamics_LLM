#!/bin/bash
# ============================================================================
# SFT Base Training - Llama 3.2 1B
# ============================================================================
set -e
export CUDA_VISIBLE_DEVICES=0
bash run_llama32_sft_base.sh
bash run_llama32_dpo_base.sh

