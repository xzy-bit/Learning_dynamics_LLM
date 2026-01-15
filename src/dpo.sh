export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
#MODEL="qwen18"
MODEL="llama3_1"
#DATASET="ultrafb"

DATASET="hh"
SFT_EPOCHS=2
N_EPOCHS=10
N_EXAMPLES=50000
EVAL_EVERY=80000
DATE=$(date +%m%d)
python -u train.py \
    loss=dpo \
    loss.beta=0.1 \
    datasets=$DATASET \
    model=$MODEL \
    exp_name="dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="sft_llama3_1_hh_ep2_0109"\
    save_ckp=true \
    eval_every=$EVAL_EVERY

python -u gen_multipt.py \
   model=${MODEL} \
   model.archive="dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}" \
   exp_name="eval_dpo_base_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
