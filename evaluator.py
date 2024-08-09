import argparse

import torch

from utils import load_task, load_model_and_tokenizer


def get_parser():
    parser = argparse.ArgumentParser(description='cosmos-eval')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--n_shots', type=int, required=False, default=None, help='Number of shots')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='Device to use')
    parser.add_argument('--limit', type=int, required=False, default=None, help='Limit the number of samples')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    model_path = args.model
    n_shots = args.n_shots
    task_name = args.task
    device = args.device
    limit = args.limit

    for arg in vars(args):
        attr = getattr(args, arg)
        if attr:
            print(f'{arg}: {attr}')

    model, tokenizer = load_model_and_tokenizer(model_path)

    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Using CPU instead.')
        device = 'cpu'

    task = load_task(task_name, n_shots)

    acc, acc_norm = task.eval_task(model, tokenizer, device, limit)

    print(f'\nacc: {acc}')
    print(f'acc_norm: {acc_norm}')
