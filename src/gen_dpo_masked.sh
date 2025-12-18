export CUDA_VISIBLE_DEVICES=3
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_mask_topk_r_0.9_k_50_s_0.0_qwen18_hh_ep4_1218\
	exp_name=eval_dpo_mask_topk_r_0.9_k_50_s_0.0_qwen18_hh_ep4_1218
