#!/bin/bash

# ================== GPU & ENV ==================
export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1

# ================== Experiment Config ==================
MODEL="qwen18"
DATASET="hh"
N_EPOCHS=1
DATE=$(date +%m%d)
N_EXAMPLES=5000
#EXP_NAME="dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
EXP_NAME="extend_sft_qwen18_hh_ep1_1202"
python analyze_entropy.py \
  model=$MODEL \
  datasets=$DATASET \
  exp_name=$EXP_NAME \
  save_path="exp_results/$EXP_NAME" \
  n_examples=$N_EXAMPLES \
  n_epochs=$N_EPOCHS

