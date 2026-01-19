import json
import torch
import argparse
import math


def main(args):
    # 1. 读取原始 HH train.jsonl
    with open(args.input_jsonl, "r") as f:
        raw_data = [json.loads(line) for line in f]

    # 2. 读取 CHES 结果
    info = torch.load(args.pt_path, map_location="cpu")

    sample_indices = info["sample_indices"]
    scores = info[args.score_key]

    assert len(sample_indices) == len(scores)

    # 3. 对齐（这是关键一步，必须和计算 CHES 时一致）
    aligned_data = [raw_data[i] for i in sample_indices]

    # 4. 排序（ln-CHES 越小越“干净”）
    order = torch.argsort(scores)  # 从小到大

    # 5. 计算要保留的样本数（比如 5%）
    if args.num_samples > 0:
        keep = args.num_samples
    else:
        keep = math.ceil(args.keep_ratio * len(order))

    selected_ids = order[:keep]

    filtered_data = [aligned_data[i] for i in selected_ids]

    # 6. 写出新的 train.jsonl
    with open(args.output_jsonl, "w") as f:
        for ex in filtered_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(filtered_data)} samples to {args.output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_jsonl", required=True,
                        help="original HH train.jsonl")
    parser.add_argument("--pt_path", required=True,
                        help="results_samples.pt")
    parser.add_argument("--output_jsonl", required=True,
                        help="filtered train.jsonl")

    parser.add_argument("--score_key", default="ln_ches_scores",
                        choices=["ches_scores", "ln_ches_scores"])

    parser.add_argument("--keep_ratio", type=float, default=0.05,
                        help="keep bottom X ratio (default: 5%)")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="override keep_ratio with exact number")

    args = parser.parse_args()
    main(args)

