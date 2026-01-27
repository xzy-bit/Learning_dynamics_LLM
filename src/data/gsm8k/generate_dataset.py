import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ======================
# Config
# ======================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SPLIT = "train"
NUM_PROMPTS = 6000
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_FILE = "gsm8k_8b.jsonl"


print("Loading GSM8K dataset...")
dataset = load_dataset("gsm8k", "main", split=SPLIT)
dataset = dataset.select(range(NUM_PROMPTS))

print(f"Loaded {len(dataset)} examples")

print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.eval()


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

results = []

for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch = dataset[i:i + BATCH_SIZE]

    prompts = [build_prompt(q) for q in batch["question"]]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.batch_decode(
        outputs[:, enc["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    for q, gold, resp in zip(batch["question"], batch["answer"], decoded):
        results.append({
            "prompt": q,
            "gold_answer": gold,
            "model_response": resp.strip(),
        })

print(f"Saving to {OUT_FILE} ...")
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Done.")