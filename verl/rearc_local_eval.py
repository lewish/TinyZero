import pandas as pd
from utils.reward_score.rearc import compute_score
from langchain_openai import ChatOpenAI

# Just a short script to get a baseline for the model and ensure it can solve at least some tasks.
# Qwen2.5-7B-Instruct gets around 3% on BARC and 1% on REARC

llm = ChatOpenAI(openai_api_base="http://127.0.0.1:1234/v1", openai_api_key="123", max_tokens=16000)

# Load the datasets
local_dir = "./datasets/barc"
train_df = pd.read_parquet(f"{local_dir}/train.parquet")
test_df = pd.read_parquet(f"{local_dir}/test.parquet")

def analyze_dataset(df):
    prompt_byte_sizes = []
    prompt_word_sizes = []
    for index, row in df.iterrows():
        prompt = row["prompt"][0]["content"]
        prompt_byte_sizes.append(len(prompt))
        prompt_word_sizes.append(len(prompt.split()))
    print(f"Average prompt byte size: {sum(prompt_byte_sizes) / len(prompt_byte_sizes)}")
    top_10_prompts = sorted(zip(prompt_byte_sizes, prompt_word_sizes), reverse=True)[:10]
    for i, (byte_size, word_size) in enumerate(top_10_prompts):
        print(f"Top {i+1} prompt - Byte size: {byte_size}, Word size: {word_size}")
        
# Evaluate the dataset
def evaluate_dataset(df):
    correct_formats = []
    correct_answers = []
    for index, row in df.iterrows():
        prompt = row["prompt"][0]["content"]
        ground_truth = row["reward_model"]["ground_truth"]
        response = llm.invoke([("human", prompt)]).content
        score = compute_score(response, ground_truth)
        correct_formats.append(score >= 0.1)
        correct_answers.append(score > 0.9)
        print(f"formats: {sum(correct_formats)} / {len(correct_formats)} answers: {sum(correct_answers)} / {len(correct_answers)}")

# Evaluate train and test datasets
analyze_dataset(train_df)
train_scores = evaluate_dataset(train_df)
# test_scores = evaluate_dataset(test_df)
