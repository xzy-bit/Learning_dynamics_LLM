python filter_hh.py \
  --input_jsonl train_dpo.jsonl \
  --pt_path ./outputs/pref_similarity/Llama-3.2-1B_hh/results_samples.pt \
  --output_jsonl train_ches_0_1.jsonl \
  --score_key ln_ches_scores \
  --keep_ratio 0.1

