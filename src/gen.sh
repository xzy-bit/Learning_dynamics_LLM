export CUDA_VISIBLE_DEVICES=1
python -u gen_inference.py \
	model=llama3_1\
	model.archive=dpo_pos_1_llama3_1_hh_ep10_0116\
	exp_name=eval_dpo_pos_1_llama3_1_hh_ep10_0116
