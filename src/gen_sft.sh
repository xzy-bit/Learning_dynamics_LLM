export CUDA_VISIBLE_DEVICES=1
python -u gen_inference_samples.py \
	model=qwen18\
	model.archive=extend_qwen18_ep4\
	exp_name=eval_extend_qwen18_ep4
