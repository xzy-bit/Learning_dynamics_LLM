export CUDA_VISIBLE_DEVICES=3
python -u gen_multipt.py \
	model=llama3_1\
	model.archive=dpo_extend_entmax_l_0.0_llama3_1_hh_ep10_0113\
	exp_name=eval_dpo_extend_entmax_l_0.0_llama3_1_hh_ep10_0113

python -u gen_multipt.py \
        model=llama3_1\
        model.archive=dpo_entmax_1.5_beta_0.25_T_0.8_usingNs_true_llama3_1_hh_ep10_0113\
        exp_name=eval_dpo_entmax_1.5_beta_0.25_T_0.8_usingNs_true_llama3_1_hh_ep10_0113
