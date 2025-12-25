export CUDA_VISIBLE_DEVICES=0
python -u gen_inference.py \
	model=qwen18\
	model.archive=dpo_asym_qwen18_hh_ep6_1225_1000000\
	exp_name=eval_dpo_asym_qwen18_hh_ep6_1225_1000000
