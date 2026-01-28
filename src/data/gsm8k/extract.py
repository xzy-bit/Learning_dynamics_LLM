import json
import re


INPUT_FILE = "gsm8k_8b.jsonl"   # 你的原始文件
OUTPUT_FILE = "gsm8k_dpo.jsonl"             # 输出 DPO 数据


# # ======================
# # 1. Answer parsers
# # ======================
#
# def extract_gold_answer(gold: str):
#     """
#     GSM8K gold answer format:
#     ...
#     #### 72
#     """
#     match = re.search(r"####\s*(-?\d+(\.\d+)?)", gold)
#     if match:
#         return match.group(1)
#     return None
#
#
# def extract_model_answer(resp: str):
#     """
#     Extract the LAST numeric value from the model response.
#     Assumes the final answer appears as the last number.
#     """
#     if resp is None:
#         return None
#
#
#     numbers = re.findall(r"-?\d+(?:\.\d+)?", resp)
#     if not numbers:
#         return None
#
#
#     return numbers[-1]
#
#
# # ======================
# # 2. Main filtering
# # ======================
#
# kept = 0
# total = 0
#
# with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
#      open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
#
#     for line in fin:
#         total += 1
#         ex = json.loads(line)
#
#         gold_ans = extract_gold_answer(ex["gold_answer"])
#         model_ans = extract_model_answer(ex["model_response"])
#         print(f"{gold_ans},{model_ans}")
#         # skip if cannot parse
#         if gold_ans is None or model_ans is None:
#             continue
#
#         if gold_ans != model_ans:
#             dpo_ex = {
#                 "prompt": ex["prompt"],
#                 "chosen": ex["gold_answer"],
#                 "rejected": ex["model_response"],
#                 # "gold_answer": gold_ans,
#                 # "model_answer": model_ans,
#             }
#             fout.write(json.dumps(dpo_ex, ensure_ascii=False) + "\n")
#             kept += 1
#
#
# print(f"Total samples: {total}")
# print(f"Kept (wrong answers): {kept}")
# print(f"Saved to: {OUTPUT_FILE}")

# with open(OUTPUT_FILE, "r", encoding="utf-8") as fout:
#     for i in range(5):
#         line = fout.readline()
#         ex = json.loads(line)
#         chosen = ex["chosen"]
#         rejected = ex["rejected"]
#         print(chosen)
#         print(rejected)

with open("train_dpo.jsonl", "w",encoding='utf-8') as train:
    with open("prob_test_gen.jsonl", "w", encoding='utf-8') as test:
        with open("gsm8k_dpo.jsonl","r",encoding='utf-8') as in_file:
            i = 0
            for line in in_file:
                if i < 3000:
                    train.write(line)
                else:
                    test.write(line)
                i += 1

with open("train_dpo.jsonl","r",encoding='utf-8') as train:
    with open("train_sft_extend.jsonl", "w", encoding='utf-8') as sft:
        for line in train:
            item = json.loads(line)
            sft_extend_item_chosen = {
                "prompt": item["prompt"],
                "chosen": item["chosen"],
            }
            sft_extend_item_rejected = {
                "prompt": item["prompt"],
                "chosen": item["rejected"],
            }
            sft.write(json.dumps(sft_extend_item_chosen)+"\n")
            sft.write(json.dumps(sft_extend_item_rejected)+"\n")
