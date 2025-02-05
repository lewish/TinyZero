import json
import os
import zipfile
import tempfile

import urllib.request
from typing import List
import random
import pandas as pd
import argparse
import yaml

parser = argparse.ArgumentParser(description="Process and save datasets.")
parser.add_argument(
    "--local_dir", type=str, required=True, help="Directory to save the datasets"
)
parser.add_argument(
    "--num_tasks", type=int, default=2000, help="Number of tasks sampled from the BARC dataset"
)
parser.add_argument(
    "--samples_per_task", type=int, default=2, help="Number of samples per task"
)
parser.add_argument(
    "--num_test_samples", type=int, default=100, help="Number of test samples"
)
parser.add_argument(
    "--output_yaml", type=bool, default=False, help="Output datasets to YAML files"
)
parser.add_argument(
    "--max_prompt_size", type=int, default=8192, help="Maximum size of the prompt"
)
args = parser.parse_args()

random.seed(42)

# URL of the zip file
url = "https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/resolve/main/data_100k.jsonl?download=true"

# Create a temporary directory
temp_dir = tempfile.gettempdir()
data_path = os.path.join(temp_dir, "barc", "data_100k.jsonl")
os.makedirs(os.path.dirname(data_path), exist_ok=True)

# Download the zip file if it doesn't exist
if not os.path.exists(data_path):
    def download_progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end="")

    urllib.request.urlretrieve(url, data_path, reporthook=download_progress_hook)
    print()  # Move to the next line after download completion


def grid_to_string(grid: list) -> str:
    return "\n".join([" ".join(map(str, row)) for row in grid])


def task_to_text_pair(problem: dict) -> dict:

    train_tasks = problem["train"]
    test_task = problem["test"]

    prompt = f"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: You are a bot that is very good at solving puzzles. Infer the pattern from the given input pairs and predict the output for the test input.
Think about the reasoning process in your mind and then provide the user with the answer.
Show your work in <think> </think> tags. Return the final answer in <answer> </answer> tags, for example:
<answer>
5 7 8 9
3 2 1 4
4 2 2 1
9 9 3 2
</answer>

"""

    prompt += "Training Examples\n"

    for i, train_task in enumerate(train_tasks):
        prompt += f"Example {i + 1}: Input\n"
        prompt += grid_to_string(train_task["input"])
        prompt += "\n\n"

        prompt += f"Example {i + 1}: Output\n"
        prompt += grid_to_string(train_task["output"])
        prompt += "\n\n"

    prompt += "Test Input:\n"
    prompt += grid_to_string(test_task["input"])

    prompt += "\n\nWhat is Output for the Test Input?\n"

    prompt += """Assistant: Let me solve this step by step.
<think>"""
    return {"prompt": prompt, "answer": grid_to_string(test_task["output"])}


barc_generations = []
with open(data_path, "r") as f:
    for i, line in enumerate(f):
        # Sample 2x as we are going to have to filter some out later
        if i >= args.num_tasks * 2:
            break
        barc_generations.append(json.loads(line))

problems = []

for row in barc_generations:
    pairs = [{ "input": pair[0], "output": pair[1]} for pair in row["examples"]]
    pairs = sorted(pairs, key=lambda x: len(x["input"]) * len(x["input"][0]), reverse=False)
    # Take the smallest quartile of pairs to keep prompts small
    pairs = pairs[0:len(pairs) // 4]
    for _ in range(args.samples_per_task):
        # Sample 4 random pairs
        random.shuffle(pairs)
        sample_pairs = pairs[:4]
        problem = {
            "train": sample_pairs[1:],
            "test": sample_pairs[0],
        }
        problems.append(task_to_text_pair(problem))


def convert_to_dataset(problems: List[dict], split: str) -> List[dict]:
    data_source = "barc"
    dataset = []
    for idx, problem in enumerate(problems):
        question = problem["prompt"]
        solution = problem["answer"]
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "pattern_recognition",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
        dataset.append(data)
    return dataset


# THIS IS OBVIOUSLY NOT A FAIR EVALUATION OF THE MODEL AS WE ARE SPLITTING SAMPLES WITHIN PROBLEMS.
# That's fine though, it's just a measure, ultimately we need to run the resulting model against the eval set which I can't be bothered to load in yet.
problems = [problem for problem in problems if len(problem["prompt"]) < args.max_prompt_size]
random.shuffle(problems)
problems = problems[: args.num_tasks]

# Convert problems to dataset format
split_index = len(problems) - args.num_test_samples
train_problems = problems[:split_index]
test_problems = problems[split_index:]

train_dataset = convert_to_dataset(train_problems, "train")
test_dataset = convert_to_dataset(test_problems, "test")

# Save datasets to parquet files
# Parse command line arguments


local_dir = args.local_dir
os.makedirs(local_dir, exist_ok=True)

train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)

train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

# Save datasets to YAML files if the flag is set
if args.output_yaml:
    train_yaml_path = os.path.join(local_dir, "train.yaml")
    test_yaml_path = os.path.join(local_dir, "test.yaml")

    with open(train_yaml_path, "w") as train_yaml_file:
        yaml.dump(train_dataset, train_yaml_file, default_flow_style=False)

    with open(test_yaml_path, "w") as test_yaml_file:
        yaml.dump(test_dataset, test_yaml_file, default_flow_style=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")
print(f"Train and test datasets saved to {local_dir}")
