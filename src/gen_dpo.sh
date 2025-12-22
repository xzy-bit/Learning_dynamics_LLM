export CUDA_VISIBLE_DEVICES=1
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_entmax_1.5_beta_1.0_usingNs_false_qwen18_hh_ep6_1222\
	exp_name=eval_dpo_entmax_1.5_beta_1.0_usingNs_false_qwen18_hh_ep6_1222
