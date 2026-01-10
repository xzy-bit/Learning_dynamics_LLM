export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1
#MODEL="pythia410m"
MODEL="llama3_1"
#MODEL="qwen18"
#DATASET="ultrafb"
DATASET="hh"
N_EPOCHS=6
N_EXAMPLES=30000
EVAL_EVERY=40000
DATE="0109"
ALPHA=1.5

#USING_NS=false
USING_NS=true
BETAS=(0.5 1.0 1.5)

for BETA in "${BETAS[@]}"; do
    python -u train.py \
    	loss=ent_dpo \
	loss.beta=0.1 \
	loss.alpha=$ALPHA \
	loss.ent_beta=$BETA\
	loss.using_ns=$USING_NS\
	datasets=$DATASET \
	model=$MODEL \
	exp_name="dpo_entmax_${ALPHA}_beta_${BETA}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
	trainer=BasicTrainer \
	using_extra_ce=false\
	n_epochs=$N_EPOCHS \
	n_examples=$N_EXAMPLES \
	save_ckp=true \
	eval_every=$EVAL_EVERY
	
done

for BETA in "${BETAS[@]}"; do
    python -u gen_multipt.py \
        model=${MODEL} \
        model.archive="dpo_entmax_${ALPHA}_beta_${BETA}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"\
        exp_name="eval_dpo_entmax_${ALPHA}_beta_${BETA}_usingNs_${USING_NS}_${MODEL}_${DATASET}_ep${N_EPOCHS}_${DATE}"
done
