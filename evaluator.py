from utils.base_utils import get_parser, load_task, load_model_and_tokenizer

from utils.metric_utils import get_metrics

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
    previous_tokens = args.previous_tokens

    model, tokenizer = load_model_and_tokenizer(model_path, device)
    task = load_task(task_name, n_shots)
    metrics = get_metrics(args, task_name)

    ret, faulty_prompts, faulty_prompts_norm = task.eval_task(model,
                                                              tokenizer,
                                                              metrics,
                                                              limit=limit,
                                                              faulty=faulty,
                                                              include_choices=include_choices,
                                                              previous_tokens=previous_tokens)

    print(f'ret: {ret}')

    if faulty:
        print(f'faulty_prompts: {faulty_prompts}')
        print(f'faulty_prompts_norm: {faulty_prompts_norm}')
