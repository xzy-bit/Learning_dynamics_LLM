import argparse
import os
from datetime import datetime

import datasets
import torch
from tqdm import tqdm

import utils.logging as logging_utils

import json
from pathlib import Path
import jsonlines
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer_and_model(model_name: str, load_model_checkpoint_from: str = "", is_lora_checkpoint: bool = False,
                             cache_dir: str = None, device=torch.device("cpu")):
    if not is_lora_checkpoint or not load_model_checkpoint_from:
        load_model_from = load_model_checkpoint_from if load_model_checkpoint_from else model_name
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=load_model_from,
            cache_dir=cache_dir,
            device_map=device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir,
            device_map=device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model=model, model_id=load_model_checkpoint_from)
        model = model.merge_and_unload()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    return tokenizer, model



PADDING_TOKEN = "<|padding|>"
PROMPT_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"


# =========================
# HH DATASET FORMAT (唯一新增)
# =========================
def __hh_helpful_create_format_input_func():
    def format_input_func(example):
        new_example = {}

        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        new_example["query"] = prompt
        new_example["text_w"] = prompt + chosen
        new_example["text_l"] = prompt + rejected
        return new_example

    return format_input_func


# =========================
# DATASET LOADING（只支持 jsonl）
# =========================
def __get_dataset(dataset_name: str, cache_dir: str = None):
    return datasets.load_dataset(
        "json",
        data_files=dataset_name,
        split="train"
    )


def __subsample_dataset(dataset, num_train_samples: int = -1, train_samples_random_seed: int = -1):
    if num_train_samples < 0:
        return torch.arange(len(dataset)), dataset

    if train_samples_random_seed > 0:
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(train_samples_random_seed))
    else:
        perm = torch.randperm(len(dataset))

    num_samples = min(num_train_samples, len(dataset))
    sample_indices = perm[:num_samples]
    dataset = dataset.select(sample_indices)
    return sample_indices, dataset


def __prepare_and_tokenize_dataset(
    sample_indices,
    dataset,
    tokenizer,
    max_input_length: int,
):
    format_input_func = __hh_helpful_create_format_input_func()

    dataset = dataset.map(format_input_func, batched=False)
    dataset = dataset.select_columns(["query", "text_w", "text_l"])

    max_input_length = max_input_length if max_input_length > 0 else None

    def tokenize_examples(example: dict):
        query_input_ids = tokenizer(
            example["query"],
            padding=False,
            truncation=max_input_length is not None,
            max_length=max_input_length,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        text_w_input_ids = tokenizer(
            example["text_w"],
            padding=False,
            truncation=max_input_length is not None,
            max_length=max_input_length,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        text_l_input_ids = tokenizer(
            example["text_l"],
            padding=False,
            truncation=max_input_length is not None,
            max_length=max_input_length,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        return {
            "query": query_input_ids,
            "text_w": text_w_input_ids,
            "text_l": text_l_input_ids,
        }

    dataset = dataset.map(tokenize_examples, batched=False)
    dataset.set_format(type="torch")

    indices_to_include = []
    for i, example in enumerate(dataset):
        query_len = example["query"][0].shape[0]
        preferred = example["text_w"][0][query_len:]
        dispreferred = example["text_l"][0][query_len:]

        if query_len == 0 or preferred.shape[0] == 0 or dispreferred.shape[0] == 0:
            continue

        indices_to_include.append(i)

    dataset = dataset.select(indices_to_include)
    return sample_indices[indices_to_include], dataset


def __update_tokenizer_setting_and_chat_tokens(tokenizer):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    if not tokenizer.eos_token:
        tokenizer.eos_token = EOS_TOKEN

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": PADDING_TOKEN})


# =========================
# UTILS（原样复制）
# =========================
def __trim_padding(input_ids, tokenizer):
    return input_ids[
        torch.argmax((input_ids != tokenizer.vocab[tokenizer.eos_token]).to(torch.int)):
    ]


def __normalized_edit_distance(seq1, seq2):
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return int(dold[-1]) / max(len(seq1), len(seq2))


def get_and_log_hidden_embedding_based_pref_similarities(logger, dataset, model, device, tokenizer):
    ches_scores = []
    ln_ches_scores = []
    last_hidden_embedding_inner_prods = []

    model.to(device)
    for example in tqdm(dataset):
        query_len = __trim_padding(example["query"][0], tokenizer).shape[0]
        preferred = __trim_padding(example["text_w"][0], tokenizer).to(device)
        dispreferred = __trim_padding(example["text_l"][0], tokenizer).to(device)

        if query_len == 0:
            continue
        if query_len == preferred.shape[0] or query_len == dispreferred.shape[0]:
            continue

        preferred_outputs = model(input_ids=preferred.unsqueeze(0), output_hidden_states=True)
        dispreferred_outputs = model(input_ids=dispreferred.unsqueeze(0), output_hidden_states=True)

        preferred_hidden = preferred_outputs.hidden_states[-1][0][query_len - 1:]
        dispreferred_hidden = dispreferred_outputs.hidden_states[-1][0][query_len - 1:]

        sum_pref = preferred_hidden.sum(dim=0)
        sum_disp = dispreferred_hidden.sum(dim=0)

        ches_scores.append(((sum_pref * sum_disp).sum() - torch.norm(sum_pref) ** 2).cpu())

        pref_len = preferred_hidden.shape[0]
        disp_len = dispreferred_hidden.shape[0]

        ln_ches_scores.append(
            ((sum_pref * sum_disp).sum() / (pref_len * disp_len)
             - torch.norm(sum_pref) ** 2 / (pref_len ** 2)).cpu()
        )

        last_hidden_embedding_inner_prods.append(
            torch.inner(preferred_hidden[-1], dispreferred_hidden[-1]).cpu()
        )

    return (
        torch.tensor(ches_scores),
        torch.tensor(ln_ches_scores),
        torch.tensor(last_hidden_embedding_inner_prods),
    )


# =========================
# MAIN（结构完全不变）
# =========================
@torch.no_grad()
def main(config: dict):
    model_name = config["model"]
    dataset_name = config["dataset"]
    num_train_samples = config["num_train_samples"]
    train_samples_random_seed = config["train_samples_random_seed"]
    max_input_length = config["max_input_length"]

    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() and config["gpu_id"] >= 0 else "cpu")

    dataset_display_name = config["custom_dataset_display_name"] if config["custom_dataset_display_name"] else "hh"
    subdir_name = model_name.split("/")[-1] + "_" + dataset_display_name

    logger = logging_utils.create_logger(
        file_logging=not config["dont_save_logs"],
        log_dir=os.path.join(config["output_dir"], subdir_name),
        log_file_name_prefix=f"log_samples_{num_train_samples}",
    )

    tokenizer, model = load_tokenizer_and_model(model_name, cache_dir=config["cache_dir"], device=device)
    __update_tokenizer_setting_and_chat_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    dataset = __get_dataset(dataset_name, cache_dir=config["cache_dir"])
    sample_indices, dataset = __subsample_dataset(dataset, num_train_samples, train_samples_random_seed)
    sample_indices, tokenized_dataset = __prepare_and_tokenize_dataset(
        sample_indices, dataset, tokenizer, max_input_length
    )

    ches_scores, ln_ches_scores, last_hidden_embedding_inner_prods = (
        get_and_log_hidden_embedding_based_pref_similarities(
            logger, tokenized_dataset, model, device, tokenizer
        )
    )

    results = {
        "sample_indices": sample_indices,
        "ches_scores": ches_scores,
        "ln_ches_scores": ln_ches_scores,
        "last_hidden_embedding_inner_prods": last_hidden_embedding_inner_prods,
    }

    torch.save(results, os.path.join(config["output_dir"], subdir_name, "results_samples.pt"))


# =========================
# CLI（完全不变）
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs/pref_similarity")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--dont_save_logs", action="store_true")
    p.add_argument("--model", type=str, default="allenai/OLMo-1B-hf")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--custom_dataset_display_name", type=str, default="")
    p.add_argument("--num_train_samples", type=int, default=-1)
    p.add_argument("--train_samples_random_seed", type=int, default=-1)
    p.add_argument("--max_input_length", type=int, default=1024)
    p.add_argument("--gpu_id", type=int, default=-1)

    args = p.parse_args()
    main(args.__dict__)

