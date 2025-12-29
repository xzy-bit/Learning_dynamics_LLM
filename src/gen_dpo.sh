export CUDA_VISIBLE_DEVICES=3
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_mask_hard_threshold_r_0.99_k_50_s_50.0_t_0.1_qwen18_hh_ep6_1229\
	exp_name=eval_dpo_mask_hard_threshold_r_0.99_k_50_s_50.0_t_0.1_qwen18_hh_ep6_1229
