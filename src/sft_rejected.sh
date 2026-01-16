export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2
MODEL="llama3_1"
#MODEL="qwen18"
#MODEL="pythia410m"
DATASET="hh"
#DATASET="ultrafb"
TRAIN_SPLIT="train_sft_rejected_only"
SFT_EPOCHS=1
N_EXAMPLES=5000
EVAL_EVERY=20000
DATE="0116"


python -u train.py \
  model=$MODEL \
  datasets=$DATASET \
  exp_name="sft_rejected_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_${DATE}" \
  trainer=BasicTrainer \
  train_split=$TRAIN_SPLIT \
  n_epochs=$SFT_EPOCHS \
  n_examples=$N_EXAMPLES \
  save_ckp="true" \
  eval_every=$EVAL_EVERY


export CUDA_LAUNCH_BLOCKING=1
SFT_EPOCHS=2
N_EPOCHS=10
N_EXAMPLES=50000
EVAL_EVERY=50000
python -u train.py \
    loss=dpo \
    loss.beta=0.1\
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_rejected_sft_${SFT_EPOCHS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="sft_rejected_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_${DATE}"\
    save_ckp=true \
    eval_every=$EVAL_EVERY

python -u gen_multipt.py \
   model=${MODEL} \
   model.archive="dpo_rejected_sft_${SFT_EPOCHS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
   exp_name="eval_dpo_rejected_sft_${SFT_EPOCHS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
