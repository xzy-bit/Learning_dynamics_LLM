export CUDA_VISIBLE_DEVICES=1
python -u gen_multipt.py \
	model=llama3_1\
	model.archive=null \
	exp_name=pretrained
