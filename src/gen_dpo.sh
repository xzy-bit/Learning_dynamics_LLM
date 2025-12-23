export CUDA_VISIBLE_DEVICES=0
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_asym_without_gradient_clip_qwen18_hh_ep6_1223\
	exp_name=eval_dpo_asym_without_gradient_clip_qwen18_hh_ep6_1223
