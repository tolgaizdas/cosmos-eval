# tasks/arc/scripts.py

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from utils import get_encoded_input, get_score


def generate_prompt(q, train_ds, n_shots=10, include_choices=False):
    prompt = ""
    random_data = train_ds.shuffle(seed=42).select(range(n_shots))

    for i, data in enumerate(random_data):
        question = data["question"]
        choices = data["choices"]["text"]
        labels = data["choices"]["label"]  # Choice labels: [A, B, C, D]
        answer = data["answerKey"]
        answer_index = ord(answer) - 65  # 'A' = 65

        prompt += f"Soru: {question}\n"
        if include_choices:
            for label, choice in zip(labels, choices):
                prompt += f"{label}. {choice}\n"
        prompt += f"Cevap: {choices[answer_index]}\n\n"

    prompt += f"Soru: {q}\n"
    prompt += "Cevap:"
    return prompt


def get_dataset():
    arc_ds = load_dataset("malhajar/arc-tr-v0.2")

    arc_train = arc_ds["test"]
    arc_valid = arc_ds["test"]

    return arc_train, arc_valid


def eval_task(model, tokenizer, train_ds, valid_ds, n_shots=25, device='cuda'):
    correct_norm, total_norm = 0, 0
    correct, total = 0, 0

    model.to(device)
    model.eval()

    for data in tqdm(valid_ds):
        question = data["question"]
        choices = data["choices"]["text"]
        labels = data["choices"]["label"]  # Choice labels: [A, B, C, D]
        answer = data["answerKey"]

        prompt = generate_prompt(question, train_ds, n_shots)
        encoded_inputs = [get_encoded_input(f"{prompt} {choice}", tokenizer, device) for choice in choices]

        scores = [get_score(model, input_tensor) for input_tensor in encoded_inputs]
        scores_norm = [score / len(choice) for score, choice in zip(scores, choices)]

        predicted_index = np.argmax(scores)
        predicted_index_norm = np.argmax(scores_norm)

        correct += int(predicted_index == labels.index(answer))
        total += 1

        correct_norm += int(predicted_index_norm == labels.index(answer))
        total_norm += 1

    acc = correct / total
    acc_norm = correct_norm / total_norm

    return acc, acc_norm
