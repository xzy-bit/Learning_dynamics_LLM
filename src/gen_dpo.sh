export CUDA_VISIBLE_DEVICES=1
python -u gen_multipt.py \
	model=qwen18\
	model.archive=dpo_base_qwen18_hh_ep6\
	exp_name=eval_dpo_base_hh_qwen18
