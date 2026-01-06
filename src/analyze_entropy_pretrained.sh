#!/bin/bash

# ================== GPU & ENV ==================
export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1

# ================== Experiment Config ==================
MODEL="qwen18"
DATASET="hh"
N_EPOCHS=1
DATE=$(date +%m%d)
N_EXAMPLES=5000
python analyze_entropy.py \
  analysis.use_pretrained=true \
  model=$MODEL \
  datasets=$DATASET \
  exp_name="dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
  save_path="exp_results/pretrained" \
  n_examples=$N_EXAMPLES \
  n_epochs=1

