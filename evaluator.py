import argparse

import torch

from utils import load_task, load_model_and_tokenizer


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


if __name__ == '__main__':
    args = get_parser().parse_args()
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr is not None:
            print(f'{arg}: {attr}')

    model_path = args.model
    n_shots = args.n_shots
    task_name = args.task
    device = args.device
    limit = args.limit
    faulty = args.print_faulty

    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Using CPU instead.')
        device = 'cpu'

    model, tokenizer = load_model_and_tokenizer(model_path)

    task = load_task(task_name, n_shots)

    metrics = ["acc", "acc_norm", "perplexity"]
    if args.exclude_acc or task_name == 'perp':
        metrics.remove("acc")
    if args.exclude_acc_norm or task_name == 'perp':
        metrics.remove("acc_norm")
    if task_name == 'perp' and (args.exclude_acc or args.exclude_acc_norm):
        print('Perplexity task does not require acc and acc_norm. Excluding them.')
    if args.exclude_perplexity:
        metrics.remove("perplexity")
    
    ret, faulty_prompts, faulty_prompts_norm = task.eval_task(model, tokenizer, device, metrics, limit, faulty)  # TODO: limit attribute can be added to Task class

    print(f'ret: {ret}')

    if faulty:
        print(f'faulty_prompts: {faulty_prompts}')
        print(f'faulty_prompts_norm: {faulty_prompts_norm}')
