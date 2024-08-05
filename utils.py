import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_encoded_input(text, tokenizer, device):
    return tokenizer.encode(text, return_tensors="pt").to(device)


def get_score(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor, labels=input_tensor)
        loss = outputs.loss.item()
    return -loss  # Lower loss better score


def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_task(task_name):
    task_name = task_name.lower().strip()
    task_module_name = f'tasks.{task_name}.scripts'

    tasks_dir = os.path.join(os.path.dirname(__file__), 'tasks')
    if tasks_dir not in sys.path:
        sys.path.append(tasks_dir)

    try:
        task_module = __import__(task_module_name, fromlist=['scripts'])
    except ImportError:
        raise ImportError(f'Could not find task {task_name}')

    print(f'Loaded task: {task_module}')
    return task_module
