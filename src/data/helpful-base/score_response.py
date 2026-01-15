import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------- clean text ----------

def extract_last_answer(text: str) -> str:
    idx = text.rfind("Assistant:")
    if idx == -1:
        return text.strip()
    return text[idx + len("Assistant:"):].strip()


def clean_answer(prompt: str, response: str) -> str:
    # 去掉 prompt continuation
    if response.startswith(prompt):
        response = response[len(prompt):]
    return extract_last_answer(response)


def clean_prompt(prompt: str) -> str:
    return prompt.replace("\n\nAssistant:", "").strip()

# ---------- main process ----------

def main(args):
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # load reward model
    tokenizer = AutoTokenizer.from_pretrained(args.rw_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.rw_model,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    device = model.device
    print(f"[INFO] Loaded reward model on {device}")

    # score the response
    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_idx, line in tqdm(enumerate(fin)):
            item = json.loads(line)

            raw_prompt = item["prompt"]
            chosen_resp = item["chosen"]
            rejected_resp = item["rejected"]

            prompt = clean_prompt(raw_prompt)
            chosen_resp = clean_answer(raw_prompt, chosen_resp)
            rejected_resp = clean_answer(raw_prompt, rejected_resp)

            rm_input_chosen = f"Human: {prompt}\nAssistant: {chosen_resp}"
            rm_input_rejected = f"Human: {prompt}\nAssistant: {rejected_resp}"


            inputs_chosen = tokenizer(
                rm_input_chosen,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            ).to(device)

            inputs_rejected = tokenizer(
                rm_input_rejected,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            ).to(device)

            with torch.no_grad():
                reward_chosen = model(**inputs_chosen).logits.squeeze().item()

            with torch.no_grad():
                reward_rejected = model(**inputs_rejected).logits.squeeze().item()

            fout.write(json.dumps({
                "prompt": prompt,
                "chosen": chosen_resp,
                "rejected": rejected_resp,
                "chosen_reward": reward_chosen,
                "rejected_reward": reward_rejected
            }, ensure_ascii=False) + "\n")


    print(f"[DONE] Saved scored results to {output_path}")


# ---------- argparse ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score prompt-response pairs with a reward model"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input jsonl (with prompt & response)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output jsonl (with prompt, response, reward)"
    )
    parser.add_argument(
        "--rw_model",
        type=str,
        help="HuggingFace reward model name"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max token length for reward model input"
    )
    args = parser.parse_args()
    main(args)
