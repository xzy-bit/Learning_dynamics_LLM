import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm


# ======================
# Config
# ======================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # 或 8B
NUM_PROMPTS = 6000
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
OUT_FILE = "gsm8k_8b.jsonl"

dataset = load_dataset("gsm8k", "main", split="train")
dataset = dataset.select(range(NUM_PROMPTS))

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_NEW_TOKENS,
)

def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

prompts = [build_prompt(q) for q in dataset["question"]]

outputs = llm.generate(prompts, sampling_params)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for ex, out in zip(dataset, outputs):
        response = out.outputs[0].text.strip()
        record = {
            "prompt": ex["question"],
            "gold_answer": ex["answer"],
            "model_response": response,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved to {OUT_FILE}")