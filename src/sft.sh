export CUDA_VISIBLE_DEVICES=1
MODEL="qwen18"
#MODEL="pythia410m"
#
DATASET="hh"
#DATASET="ultrafb"
TRAINER="BasicTrainer"
TRAIN_SPLIT="train_dpo"
N_EPOCHS=4
N_EXAMPLES=20000
EVAL_EVERY=40000

python -u train.py \
  model=$MODEL \
  exp_name="sft_${MODEL}_${DATASET}_ep${N_EPOCHS}" \
  datasets=$DATASET \
  trainer="BasicTrainer" \
  train_split="train_dpo" \
  n_epochs=$N_EPOCHS \
  n_examples=$N_EXAMPLES \
  save_ckp="true" \
  eval_every=$EVAL_EVERY
