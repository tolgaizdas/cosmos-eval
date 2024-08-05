import argparse

import torch

from utils import load_task, load_model_and_tokenizer


def get_parser():
    parser = argparse.ArgumentParser(description='cosmos-eval')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model')
    # parser.add_argument('-t', '--tokenizer', type=str, required=True, help='Path to the tokenizer')
    # parser.add_argument('--train_ds', type=str, required=True, help='Path to the training dataset')
    # parser.add_argument('--valid_ds', type=str, required=True, help='Path to the validation dataset')
    parser.add_argument('--n_shots', type=int, default=0, help='Number of shots')
    # parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate')

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    model_path = args.model
    n_shots = args.n_shots
    task_name = args.task

    print(f'model_path: {model_path}')
    print(f'n_shots: {n_shots}')
    print(f'task_name: {task_name}')

    model, tokenizer = load_model_and_tokenizer(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    task = load_task(task_name)
    train_ds, valid_ds = task.get_dataset()

    acc, acc_norm = task.eval_task(model, tokenizer, train_ds, valid_ds, n_shots, device)
    print(f'Accuracy: {acc}')
