export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2
MODEL="qwen18"

#MODEL="pythia410m"
DATASET="hh"
#DATASET="ultrafb"
TRAIN_SPLIT="train_sft_extend"
N_EPOCHS=1
N_EXAMPLES=10000
EVAL_EVERY=20000

python -u train.py \
  model=$MODEL \
  datasets=$DATASET \
  exp_name="extend_sft_${MODEL}_${DATASET}_ep${N_EPOCHS}" \
  trainer=BasicTrainer \
  train_split=train_sft_extend \
  n_epochs=$N_EPOCHS \
  n_examples=$N_EXAMPLES \
  save_ckp="true" \
  eval_every=$EVAL_EVERY
