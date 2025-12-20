export CUDA_VISIBLE_DEVICES=1
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_mask_hard_threshold_r_0.9_k_50_s_0.0_t_0.01_qwen18_hh_ep4_1220\
	exp_name=eval_dpo_mask_hard_threshold_r_0.9_k_50_s_0.0_t_0.01_qwen18_hh_ep4_1220
