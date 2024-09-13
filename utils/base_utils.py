import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser(description='cosmos-eval')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate')
    parser.add_argument('--n_shots', type=int, required=False, default=None, help='Number of shots')
    parser.add_argument('--device', type=str, required=False, default='cuda', help='Device to use')
    parser.add_argument('--limit', type=int, required=False, default=None, help='Limit the number of samples')
    parser.add_argument('--print-faulty', action='store_true', help='Print faulty prompts')
    parser.add_argument('--exclude-acc', action='store_true', help='Exclude accuracy')
    parser.add_argument('--exclude-acc-norm', action='store_true', help='Exclude normalized accuracy')
    parser.add_argument('--exclude-perplexity', action='store_true', help='Exclude perplexity')
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


def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_task(task_name, n_shots):
    task_name = task_name.lower().strip()
    if task_name == 'hellaswag':
        from tasks.hellaswag.hellaswag import Hellaswag
        task = Hellaswag(n_shots) if n_shots is not None else Hellaswag()  # TODO: default n_shots can be handled in the task class
    elif task_name == 'arc':
        from tasks.arc.arc import ARC
        task = ARC(n_shots) if n_shots is not None else ARC()
    elif task_name == 'teog':
        from tasks.teog.teog import TEOG
        task = TEOG(n_shots) if n_shots is not None else TEOG()
    elif task_name == 'perp':
        from tasks.perp.perp import Perp
        task = Perp(n_shots) if n_shots is not None else Perp()
    elif task_name == 'openbookqa':
        from tasks.openbookqa.openbookqa import OpenBookQA
        task = OpenBookQA(n_shots) if n_shots is not None else OpenBookQA()
    elif task_name == 'xstorycloze':
        from tasks.xstorycloze.xstorycloze import XStoryCloze
        task = XStoryCloze(n_shots) if n_shots is not None else XStoryCloze()
    else:
        raise ValueError(f'Unknown task: {task_name}')

    if n_shots is None:
        print(f'n_shots is not provided. Using default value: {task.n_shots}')

    return task
