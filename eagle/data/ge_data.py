import json
from datasets import load_dataset
import os
import argparse

# python -m eagle.data.ge_data --input ../datasets/mt_bench_human_judgments --output data/mt-bench-judge/train.jsonl
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="~/code/datasets/mt_bench_human_judgments", type=str)
parser.add_argument("--output", default="question.jsonl", type=str)
parser.add_argument("--role", default="user", type=str)
args = parser.parse_args()

dataset = load_dataset(args.input, split="human")

def extract_turns(sample):
    """
    Extracts turns from the sample based on the specified role.
    """
    return {"turns": [x["content"] for x in sample["conversation_a"] if x["role"] == "user"]}

ds = dataset.map(extract_turns)

remove_list = ['model_a', 'model_b', 'winner', 'judge', 'conversation_a', 'conversation_b', 'turn']
for key in remove_list:
    if key in ds.column_names:
        ds = ds.remove_columns(key)

output_file = args.output

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "a", encoding="utf-8") as f:
    for sample in ds:
        # Ensure unique IDs
        # if sample["question_id"] in id_set:
        #     continue
        # id_set.add(sample["question_id"])
        f.write(json.dumps(sample) + "\n")

print(f"Output to: {output_file}")

