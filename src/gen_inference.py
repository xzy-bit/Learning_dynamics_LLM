import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
import os
import json
import hydra
from typing import Set
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import torch.nn.functional as F
from entmax import sparsemax

from utils import (
    get_local_dir,
    get_local_run_dir,
    disable_dropout,
)
from preference_datasets import get_batch_iterator
OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
)

# =====================================================
# inference for ONE split (等价 evaluation_get_response)
# =====================================================
def run_inference_one_split(
    policy,
    tokenizer,
    config: DictConfig,
    prob_set: str,
):
    data_iterator_kwargs = dict(
        names=config.datasets,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    iterator = get_batch_iterator(
        **data_iterator_kwargs,
        split=prob_set,
        n_examples=500,                     # 与 Trainer 完全一致
        shuffle=False,
        batch_size=config.eval_batch_size,
        silent=True,
    )

    policy.eval()
    all_samples = []

    decode_act = "sparsemax"
    device = next(policy.parameters()).device

    with torch.no_grad():
        for batch in tqdm(iterator, desc=f"Generating {prob_set}"):
            input_ids = batch["prompt_input_ids"].to(device)
            attn_mask = batch["prompt_attention_mask"].to(device)

            # ===== 协议级 assert =====
            assert input_ids.shape[1] <= config.max_prompt_length

            generated = input_ids
            mask = attn_mask

            prompt_len = input_ids.shape[1]
            max_new_tokens = config.max_length - prompt_len
            assert max_new_tokens > 0

            finished = torch.zeros(
                input_ids.size(0), dtype=torch.bool, device=device
            )

            for _ in range(max_new_tokens):
                outputs = policy(input_ids=generated, attention_mask=mask)
                logits = outputs.logits[:, -1, :]

                if decode_act == "softmax":
                    probs = F.softmax(logits, dim=-1)
                elif decode_act == "sparsemax":
                    probs = sparsemax(logits, dim=-1)
                else:
                    raise ValueError(f"Unknown decode_activation: {decode_act}")

                next_token = torch.multinomial(probs, 1).squeeze(1)

                generated = torch.cat([generated, next_token[:, None]], dim=-1)
                mask = torch.cat(
                    [mask, torch.ones_like(next_token[:, None])], dim=-1
                )

                finished |= (next_token == tokenizer.eos_token_id)
                if finished.all():
                    break

            # ===== 长度协议 assert =====
            assert generated.shape[1] <= config.max_length

            texts = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

            for p, t in zip(batch["prompt"], texts):
                all_samples.append({
                    "prompt": p,
                    "response": t,
                })

    output_path = os.path.join(
        config.save_path,
        f"{prob_set}_response.jsonl"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[OK] {prob_set}: wrote {len(all_samples)} samples → {output_path}")
    return len(all_samples)


# =====================================================
# Hydra main
# =====================================================
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):

    # ---------- resolve & check ----------
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    print(OmegaConf.to_yaml(config))

    # ---------- 固定随机种子（关键） ----------
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # ---------- output root ----------
    gen_root = os.path.join("exp_results", config.exp_name)
    os.makedirs(gen_root, exist_ok=True)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)

    # ---------- build policy once ----------
    print("building policy")
    policy_dtype = getattr(torch, config.model.policy_dtype)

    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        torch_dtype=policy_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    disable_dropout(policy)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ---------- locate checkpoint dir ----------
    if config.model.archive is None:
        raise ValueError("config.model.archive must be set for multi-ckpt inference")

    model_dir = os.path.join("exp_results", config.model.archive)
    if not os.path.isdir(model_dir):
        raise ValueError(f"Checkpoint dir not found: {model_dir}")

    # ---------- find checkpoints (仿照你原逻辑) ----------
    ckpt_map = {}
    for i in range(1, 6):
        ckpt_map[f"policy_{i}.pt"] = f"{config.exp_name}_ep{i}"
    ckpt_map["policy.pt"] = f"{config.exp_name}_ep6"

    print("Checkpoint map:")
    for k, v in ckpt_map.items():
        print(f"  {k} -> {v}")

    # ---------- loop over checkpoints ----------
    for ckpt_name, exp_dir in ckpt_map.items():
        load_path = os.path.join(model_dir, ckpt_name)
        if not os.path.exists(load_path):
            print(f"[WARN] checkpoint not found: {load_path}")
            continue

        print(f"\n===== Loading checkpoint: {ckpt_name} =====")
        state_dict = torch.load(load_path, map_location="cpu")
        policy.load_state_dict(state_dict["state"])
        policy.eval()
        torch.cuda.empty_cache()

        # 每个 ckpt 单独一个输出目录
        config.save_path = os.path.join(gen_root, exp_dir)
        os.makedirs(config.save_path, exist_ok=True)
        print(f"Saving inference outputs to {config.save_path}")

        # 完全复刻 Trainer 的两次调用
        n1 = run_inference_one_split(policy, tokenizer, config, "prob_train_gen")
        n2 = run_inference_one_split(policy, tokenizer, config, "prob_test_gen")

        assert n1 == 500 and n2 == 500, f"Unexpected counts: {n1}, {n2}"
        print(f"[DONE] {ckpt_name}: total samples = {n1 + n2}")

    print("\nAll checkpoints finished.")


if __name__ == "__main__":
    main()
