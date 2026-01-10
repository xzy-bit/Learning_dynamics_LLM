export CUDA_VISIBLE_DEVICES=0
MODEL="llama3_1"
#MODEL="qwen18"
#MODEL="pythia410m"
DATASET="hh"
#DATASET="ultrafb"
TRAINER="BasicTrainer"
TRAIN_SPLIT="train_dpo"
N_EPOCHS=2
N_EXAMPLES=10000
EVAL_EVERY=20000
DATE="0109"
:<<sft
python -u train.py \
  model=$MODEL \
  exp_name="sft_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
  datasets=$DATASET \
  trainer="BasicTrainer" \
  train_split="train_dpo" \
  n_epochs=$N_EPOCHS \
  n_examples=$N_EXAMPLES \
  save_ckp="true" \
  eval_every=$EVAL_EVERY
sft

export CUDA_LAUNCH_BLOCKING=1
SFT_EPOCHS=2
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=50000
python -u train.py \
    loss=dpo \
    loss.beta=0.1 \
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="sft_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_${DATE}"\
    save_ckp=true \
    eval_every=$EVAL_EVERY

python -u gen_multipt.py \
   model=${MODEL} \
   model.archive="dpo_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
   exp_name="eval_dpo_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"

#bash score.sh
