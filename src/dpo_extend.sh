export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
MODEL="llama3_1"
#MODEL="pythia410m"
#MODEL="qwen18"
#DATASET="ultrafb"
DATASET="hh"
SFT_EPOCHS=1
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=100000
DATE="0113"
LAMBDAS=(0.0 0.1)

for LAMBDA in "${LAMBDAS[@]}"; do
python -u train.py \
    loss=ent_dpo \
    loss.beta=0.1 \
	  loss.alpha=$ALPHA \
	  loss.ent_beta=$BETA\
	  loss.using_ns=$USING_NS\
	  loss.temperature=$TEMP\
	  using_extra_ce=true \
    ce_lambda=$LAMBDA \
    model=$MODEL \
    datasets=$DATASET \
    exp_name="dpo_extend_entmax_l_${LAMBDA}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
    trainer=BasicTrainer \
    n_epochs=$N_EPOCHS \
    n_examples=$N_EXAMPLES \
    model.archive="extend_sft_${MODEL}_${DATASET}_ep${SFT_EPOCHS}_1202" \
    save_ckp=true \
    eval_every=$EVAL_EVERY

python -u gen_multipt.py \
   model=${MODEL} \
   model.archive="dpo_extend_entmax_l_${LAMBDA}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
   exp_name="eval_dpo_extend_entmax_l_${LAMBDA}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}}"

