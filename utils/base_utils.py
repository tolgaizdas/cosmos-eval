import argparse
import importlib

from tabulate import tabulate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser(description='cosmos-eval')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate')
    parser.add_argument('--n-shots', type=int, required=False, default=None, help='Number of shots')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='Device to use')
    parser.add_argument('--limit', type=int, required=False, default=None, help='Limit the number of samples')
    parser.add_argument('--previous-token-generator', type=str, required=False, default=None, help='Model for generating previous tokens')
    parser.add_argument('--previous-explicit-tokenizer', type=str, required=False, default=None, help='Explicit tokenizer for previous token generator')
    parser.add_argument('--previous-conf', type=int, choices=[0, 1], required=False, default=0, help="Define the configuration for previous token generation: 0 or 1.")
    parser.add_argument('--explicit-tokenizer', type=str, required=False, default=None, help='Explicit tokenizer to use')
    parser.add_argument('--from-tf', action='store_true', help='Load model from TensorFlow')
    parser.add_argument('--previous-from-tf', action='store_true', help='Load previous token generator from TensorFlow')
    parser.add_argument('--print-faulty', action='store_true', help='Print faulty prompts')
    parser.add_argument('--include-choices-in-prompt', action='store_true', help='Include choices in prompt')
    parser.add_argument('--exclude-acc', action='store_true', help='Exclude accuracy')
    parser.add_argument('--exclude-acc-norm', action='store_true', help='Exclude normalized accuracy')
    parser.add_argument('--exclude-perplexity', action='store_true', help='Exclude perplexity')
    return parser


def load_model_and_tokenizer(model_name, device, from_tf=False, explicit_tokenizer=None):
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Using CPU instead.')
        device = 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_name, from_tf=from_tf).to(device)
    tokenizer = AutoTokenizer.from_pretrained(explicit_tokenizer if explicit_tokenizer is not None else model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_task(task_name, n_shots):
    TASKS = {
        'hellaswag': 'tasks.Hellaswag',
        'arc': 'tasks.ARC',
        'teog': 'tasks.TEOG',
        'perp': 'tasks.Perp',
        'openbookqa': 'tasks.OpenBookQA',
        'xstorycloze': 'tasks.XStoryCloze',
        'race': 'tasks.Race',
        'xcopa': 'tasks.XCOPA'
    }

    task_name = task_name.lower().strip()
    task_path = TASKS.get(task_name)

    assert task_path is not None, f"Unknown task: '{task_name}'. Available tasks: {[key for key in TASKS.keys()]}"

    module_name, class_name = task_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    task_class = getattr(module, class_name)

    task = task_class(n_shots) if n_shots is not None else task_class()

    if n_shots is None:
        print(f'n_shots is not provided. Using default value: {task.n_shots}')

    return task


def print_results(model_path, task_name, n_shots, limit, ret):
    table_data = [
        ["model", model_path],
        ["task", task_name],
        ["few-shots", n_shots],
    ]

    if limit is not None:
        table_data.append(["limit", limit])

    for metric, value in ret.items():
        if value is not None and "rel_" not in metric:
            table_data.append([metric, f"{value:.2f}"])
            if f"rel_{metric}" in ret:
                table_data.append([f"rel_{metric}", f"{ret[f'rel_{metric}']:.2f}"])

    # Print the results in horizontal table format
    print("\nResults:")
    transposed_table = list(zip(*table_data))
    print(tabulate(transposed_table, headers="firstrow", tablefmt="grid"))
