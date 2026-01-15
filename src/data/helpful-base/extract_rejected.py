import json

input_file = "train_dpo.jsonl"
output_file = "train_sft_rejected_only.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        new_data = {
            "prompt": data["prompt"],
            "chosen": data["rejected"]
        }
        fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")

