import json

input_path = "filtered_dpo_reward.jsonl"     # 原 preference 数据
output_path = "filtered_dpo.jsonl"  # 输出的 SFT 数据

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    for line in fin:
        data = json.loads(line)

        prompt = data["prompt"]

        # 原 chosen → SFT
        fout.write(json.dumps({
            "prompt": prompt,
            "chosen": data["chosen"],
            "rejected":data["rejected"]
        }, ensure_ascii=False) + "\n")

        ## 原 rejected → 也当作 SFT
        #fout.write(json.dumps({
        #    "prompt": prompt,
        #    "chosen": data["rejected"]
        #}, ensure_ascii=False) + "\n")

print("✅ extend-SFT 数据生成完成")

