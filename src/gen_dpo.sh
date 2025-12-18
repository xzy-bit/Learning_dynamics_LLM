export CUDA_VISIBLE_DEVICES=0
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_mask_sparsemax_r_0.1_k_3_s_0.0_qwen18_hh_ep4_1217\
	exp_name=eval_dpo_mask_sparsemax_r_0.1_k_3_s_0.0_qwen18_hh_ep4_1217
