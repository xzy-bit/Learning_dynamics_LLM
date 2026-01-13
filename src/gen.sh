export CUDA_VISIBLE_DEVICES=1
python -u gen_inference.py \
	model=llama3_1\
	model.archive=sft_llama3_1_hh_ep2_0109\
	exp_name=eval_sft_llama3_1_hh_ep2_0109
