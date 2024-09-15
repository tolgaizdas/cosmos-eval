import torch

from utils.base_utils import get_parser, get_metrics, load_task, load_model_and_tokenizer

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
    include_choices = args.include_choices_in_prompt

    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Using CPU instead.')
        device = 'cpu'

    model, tokenizer = load_model_and_tokenizer(model_path)
    task = load_task(task_name, n_shots)
    metrics = get_metrics(args, task_name)

    ret, faulty_prompts, faulty_prompts_norm = task.eval_task(model,
                                                              tokenizer,
                                                              device,
                                                              metrics,
                                                              limit,
                                                              faulty,
                                                              include_choices)

    print(f'ret: {ret}')

    if faulty:
        print(f'faulty_prompts: {faulty_prompts}')
        print(f'faulty_prompts_norm: {faulty_prompts_norm}')
