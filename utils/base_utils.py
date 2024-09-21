import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser(description='cosmos-eval')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate')
    parser.add_argument('--n_shots', type=int, required=False, default=None, help='Number of shots')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='Device to use')
    parser.add_argument('--limit', type=int, required=False, default=None, help='Limit the number of samples')
    parser.add_argument('--print-faulty', action='store_true', help='Print faulty prompts')
    parser.add_argument('--include-choices-in-prompt', action='store_true', help='Include choices in prompt')
    parser.add_argument('--exclude-acc', action='store_true', help='Exclude accuracy')
    parser.add_argument('--exclude-acc-norm', action='store_true', help='Exclude normalized accuracy')
    parser.add_argument('--exclude-perplexity', action='store_true', help='Exclude perplexity')
    parser.add_argument('--previous-tokens', action='store_true', help='Generate previous tokens for prompt')
    return parser


def get_metrics(args, task_name):
    metrics = ["acc", "acc_norm", "perplexity"]
    if args.exclude_acc or task_name == 'perp':
        metrics.remove("acc")
    if args.exclude_acc_norm or task_name == 'perp':
        metrics.remove("acc_norm")
    if task_name == 'perp' and (args.exclude_acc or args.exclude_acc_norm):
        print('Perplexity task does not require acc and acc_norm. Excluding them.')
    if args.exclude_perplexity:
        metrics.remove("perplexity")
    return metrics


def load_model_and_tokenizer(model_name, device):
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Using CPU instead.')
        device = 'cpu'

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_task(task_name, n_shots):
    task_name = task_name.lower().strip()
    if task_name == 'hellaswag':
        from tasks import Hellaswag
        task = Hellaswag(n_shots) if n_shots is not None else Hellaswag()  # TODO: default n_shots can be handled in the task class
    elif task_name == 'arc':
        from tasks import ARC
        task = ARC(n_shots) if n_shots is not None else ARC()
    elif task_name == 'teog':
        from tasks import TEOG
        task = TEOG(n_shots) if n_shots is not None else TEOG()
    elif task_name == 'perp':
        from tasks import Perp
        task = Perp(n_shots) if n_shots is not None else Perp()
    elif task_name == 'openbookqa':
        from tasks import OpenBookQA
        task = OpenBookQA(n_shots) if n_shots is not None else OpenBookQA()
    elif task_name == 'xstorycloze':
        from tasks import XStoryCloze
        task = XStoryCloze(n_shots) if n_shots is not None else XStoryCloze()
    elif task_name == 'race':
        from tasks import Race
        task = Race(n_shots) if n_shots is not None else Race()
    else:
        raise ValueError(f'Unknown task: {task_name}')

    if n_shots is None:
        print(f'n_shots is not provided. Using default value: {task.n_shots}')

    return task
