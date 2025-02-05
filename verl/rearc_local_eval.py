import pandas as pd
from utils.reward_score.rearc import compute_score
from langchain_openai import ChatOpenAI

# Just a short script to get a baseline for the model and ensure it can solve at least some tasks.

llm = ChatOpenAI(openai_api_base="http://127.0.0.1:1234/v1", openai_api_key="123", max_tokens=16000)

# Load the datasets
local_dir = "./datasets/barc"
train_df = pd.read_parquet(f"{local_dir}/train.parquet")
test_df = pd.read_parquet(f"{local_dir}/test.parquet")

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
train_scores = evaluate_dataset(train_df)
# test_scores = evaluate_dataset(test_df)
