export CUDA_VISIBLE_DEVICES=3
export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
MODEL="llama3_1"
#MODEL="qwen18"
#DATASET="ultrafb"
DATASET="hh"
N_EPOCHS=10
N_EXAMPLES=50000
EVAL_EVERY=80000
DATE="0113"

USING_NS=true

ALPHA=1.5
BETA=0.25
TEMPS=(1.0)
LAMBDA=0.1

for TEMP in "${TEMPS[@]}"; do
    python -u train.py \
    	loss=ent_dpo \
	loss.beta=0.1 \
	loss.alpha=$ALPHA \
	loss.ent_beta=$BETA\
	loss.using_ns=$USING_NS\
	loss.temperature=$TEMP\
	using_extra_ce=false \
  	ce_lambda=$LAMBDA \
	datasets=$DATASET \
	model=$MODEL \
	exp_name="dpo_entmax_${ALPHA}_beta_${BETA}_T_${TEMP}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
	trainer=BasicTrainer \
	using_extra_ce=false\
	n_epochs=$N_EPOCHS \
	n_examples=$N_EXAMPLES \
	save_ckp=true \
	eval_every=$EVAL_EVERY
	
done

for TEMP in "${TEMPS[@]}"; do
    python -u gen_multipt.py \
        model=${MODEL} \
        model.archive="dpo_entmax_${ALPHA}_beta_${BETA}_T_${TEMP}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
        exp_name="eval_dpo_entmax_${ALPHA}_beta_${BETA}_T_${TEMP}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
done
bash score.sh
