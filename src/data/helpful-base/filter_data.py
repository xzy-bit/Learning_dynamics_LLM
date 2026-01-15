import json

input_file = "train_dpo_reward.jsonl"
output_file = "filtered_dpo_reward.jsonl"

min_gap = 10
max_gap = 0.2

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        chosen_reward = data["chosen_reward"]
        rejected_reward = data["rejected_reward"]

        # if chosen_reward-rejected_reward >0.2 and chosen_reward-rejected_reward <min_gap:
        #     min_gap = min(chosen_reward-rejected_reward, min_gap)
        #
        # if chosen_reward - rejected_reward >max_gap:
        #     max_gap = max(chosen_reward-rejected_reward, max_gap)
        if chosen_reward - rejected_reward < 0:
            continue

        new_data = {
            "prompt": data["prompt"],
            "chosen": data["chosen"],
            "rejected": data["rejected"],
            "rejected_reward": rejected_reward,
            "chosen_reward": chosen_reward,
        }
        fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")

