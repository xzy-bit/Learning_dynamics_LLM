export CUDA_VISIBLE_DEVICES=2
python -u gen_multipt.py \
	model=llama3_1\
	model.archive=dpo_extend_entmax_l_0.01_llama3_1_hh_ep10_0113\
	exp_name=eval_dpo_extend_entmax_l_0.01_llama3_1_hh_ep10_0113

