"""Compute the speedup of generated answers over an autoregressive baseline."""

import argparse
import json
from statistics import fmean

from transformers import AutoTokenizer


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def speculative_speeds(records):
    speeds = []
    for record in records:
        choice = record["choices"][0]
        tokens = sum(choice["new_tokens"])
        wall_time = sum(choice["wall_time"])
        speeds.append(tokens / wall_time)
    return speeds


def baseline_speeds(records, tokenizer):
    speeds = []
    for record in records:
        choice = record["choices"][0]
        tokens = sum(len(tokenizer(turn).input_ids) - 1 for turn in choice["turns"])
        wall_time = sum(choice["wall_time"])
        speeds.append(tokens / wall_time)
    return speeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", required=True, help="Hugging Face model ID or local tokenizer path.")
    parser.add_argument("--answer-file", required=True, help="RADAR or EAGLE answer JSONL file.")
    parser.add_argument("--baseline-answer-file", required=True, help="Autoregressive baseline answer JSONL file.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    candidate_speed = fmean(speculative_speeds(load_jsonl(args.answer_file)))
    baseline_speed = fmean(baseline_speeds(load_jsonl(args.baseline_answer_file), tokenizer))

    print(f"candidate tokens/s: {candidate_speed:.4f}")
    print(f"baseline tokens/s: {baseline_speed:.4f}")
    print(f"speedup: {candidate_speed / baseline_speed:.4f}x")


if __name__ == "__main__":
    main()
