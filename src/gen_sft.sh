export CUDA_VISIBLE_DEVICES=3
python -u gen_inference_samples.py \
	model=qwen18\
	model.archive=sft_qwen18_hh_ep2_1202\
	exp_name=eval_sft_qwen18_hh_ep2_1202

python -u gen_inference_samples.py \
        model=qwen18\
        model.archive=extend_sft_qwen18_hh_ep1_1202 \
        exp_name=eval_extend_sft_qwen18_hh_ep1_1202
