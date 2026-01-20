export CUDA_VISIBLE_DEVICES=3
MODEL="llama3_1"
#MODEL="qwen18"
#MODEL="pythia410m"
DATASET="hh"
#DATASET="ultrafb"
TRAINER="BasicTrainer"
TRAIN_SPLIT="train_dpo"
SFT_EPOCHS=2
N_EXAMPLES=1000
EVAL_EVERY=20000
DATE="0116"


python -u train.py \
  model=$MODEL \
  exp_name="sft_ches_ratio_0_1_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_${DATE}" \
  datasets=$DATASET \
  trainer="BasicTrainer" \
  train_split="train_ches_0_1" \
  n_epochs=$SFT_EPOCHS \
  n_examples=$N_EXAMPLES \
  save_ckp="true" \
  eval_every=$EVAL_EVERY


export CUDA_LAUNCH_BLOCKING=1
N_EPOCHS=10
N_EXAMPLES=5000
EVAL_EVERY=50000

python -u train.py \
    loss=dpo \
    loss.beta=0.1 \
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_ches_ratio_0_1_${SFT_EPOCHS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    train_split="train_ches_0_1" \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="sft_ches_ratio_0_1_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_${DATE}"\
    save_ckp=true \
    eval_every=$EVAL_EVERY

python -u gen_multipt.py \
   model=${MODEL} \
   model.archive="dpo_ches_ratio_0_1_${SFT_EPOCHS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
   exp_name="eval_dpo_ches_ratio_0_1_${SFT_EPOCHS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"

bash score.sh
