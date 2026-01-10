export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
MODEL="llama3_1"
#MODEL="qwen18"
#MODEL="pythia410m"
DATASET="hh"
#DATASET="ultrafb"
TRAIN_SPLIT="train_sft_extend"
N_EPOCHS=1
N_EXAMPLES=10000
EVAL_EVERY=20000
DATE="0109"

:<<sft
python -u train.py \
  model=$MODEL \
  datasets=$DATASET \
  exp_name="extend_sft_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
  trainer=BasicTrainer \
  train_split=train_sft_extend \
  n_epochs=$N_EPOCHS \
  n_examples=$N_EXAMPLES \
  save_ckp="true" \
  eval_every=$EVAL_EVERY
sft

export CUDA_LAUNCH_BLOCKING=1
SFT_EPOCHS=1
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=50000
python -u train.py \
    loss=ent_dpo \
    loss.beta=0.1 \
    loss.alpha=1.5 \
    loss.ent_beta=1.0\
    loss.using_ns=true\
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_extend_entmax_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="extend_sft_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_${DATE}"\
    save_ckp=true \
    eval_every=$EVAL_EVERY

python -u gen_multipt.py \
   model=${MODEL} \
   model.archive="dpo_extend_entmax_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
   exp_name="eval_dpo_extend_entmax_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
