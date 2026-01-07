#!/bin/bash

# ================== GPU & ENV ==================
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1

# ================== Experiment Config ==================
MODEL="qwen18"
DATASET="hh"
N_EPOCHS=6
DATE=$(date +%m%d)
N_EXAMPLES=5000
#EXP_NAME="dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_0105"
#EXP_NAME="extend_sft_qwen18_hh_ep1_1202"
EXP_NAME="sft_qwen18_hh_ep2_1202"
#EXP_NAME="dpo_extend_qwen18_hh_ep6_0105"
python analyze_entropy.py \
  analysis.mode="tf"\
  model=$MODEL \
  datasets=$DATASET \
  exp_name=$EXP_NAME \
  save_path="exp_results/$EXP_NAME" \
  n_examples=$N_EXAMPLES \
  n_epochs=6

