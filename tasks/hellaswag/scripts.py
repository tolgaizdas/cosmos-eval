# tasks/hellaswag/scripts.py

import re

from datasets import load_dataset
from tqdm import tqdm

from utils import get_encoded_input, get_score


def preprocess(text):
    text = text.replace(" [title]", ". ")
    text = text.replace(" [header]", ". ")
    text = text.replace(" [başlık]", ". ")
    text = text.replace(" [adım]", ". ")
    text = text.replace("  ", " ")
    text = text.replace("..", ".")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.strip()
    return text


def process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    out_doc = {"ctx": preprocess(ctx), "endings": [preprocess(ending) for ending in doc["endings"]],
               "label": int(doc["label"]), }
    return out_doc


def get_dataset():
    hellaswag = load_dataset("malhajar/hellaswag-tr")
    hellaswag_valid_v02 = load_dataset("malhajar/hellaswag_tr-v0.2", split="validation")

    hellaswag_train = hellaswag["train"]
    hellaswag_valid = hellaswag_valid_v02  # 8k
    # hellaswag_valid = concatenate_datasets([hellaswag_valid_v02, hellaswag["validation"]]) #18k

    hellaswag_train = hellaswag_train.map(process_doc)
    hellaswag_valid = hellaswag_valid.map(process_doc)

    return hellaswag_train, hellaswag_valid


def generate_prompt(ctx, train_ds, n_shots=10):
    prompt = ""
    random_data = train_ds.shuffle(seed=42).select(range(n_shots))

    for i, data in enumerate(random_data):
        context = data["ctx"]
        endings = data["endings"]
        answer = data["label"]

        prompt += f"İçerik: {context}\n"
        """
        for i in range(len(endings)):
            prompt += f"{chr(i+65)}. {endings[i]}\n"
        """
        prompt += f"Cevap: {endings[answer]}\n\n"

    prompt += f"İçerik: {ctx}\n"
    prompt += "Cevap:"
    return prompt


def eval_task(model, tokenizer, train_ds, valid_ds, n_shots=10, device='cuda'):
    correct_norm, total_norm = 0, 0
    correct, total = 0, 0

    model.to(device)
    model.eval()

    for data in tqdm(valid_ds):
        context = data["ctx"]
        endings = data["endings"]
        label = data["label"]

        prompt = generate_prompt(context, train_ds, n_shots)
        encoded_inputs = [get_encoded_input(f"{prompt} {ending}", tokenizer, device) for ending in endings]

        scores = [get_score(model, input_tensor) for input_tensor in encoded_inputs]
        scores_norm = [score / len(ending) for score, ending in zip(scores, endings)]

        predicted_index = scores.index(max(scores))
        predicted_index_norm = scores_norm.index(max(scores_norm))

        correct += int(predicted_index == label)
        total += 1

        correct_norm += int(predicted_index_norm == label)
        total_norm += 1

    acc = correct / total
    acc_norm = correct_norm / total_norm

    return acc, acc_norm
