from utils.base_utils import get_parser, load_task, load_model_and_tokenizer, print_results

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
    previous_token_generator = args.previous_token_generator
    from_tf = args.from_tf
    explicit_tokenizer = args.explicit_tokenizer

    previous_tokens = previous_token_generator is not None  # If previous_token_generator is provided, use previous tokens

    model, tokenizer = load_model_and_tokenizer(model_path, device, from_tf=from_tf, explicit_tokenizer=explicit_tokenizer)

    task = load_task(task_name, n_shots)
    if previous_tokens:
        task.prompt_generator.model, task.prompt_generator.tokenizer = load_model_and_tokenizer(previous_token_generator, device)

    metrics = get_metrics(args, task_name)

    ret, faulty_prompts, faulty_prompts_norm = task.eval_task(model,
                                                              tokenizer,
                                                              metrics,
                                                              limit=limit,
                                                              faulty=faulty,
                                                              include_choices=include_choices,
                                                              previous_tokens=previous_tokens)


    print_results(model_path, task_name, n_shots, limit, ret)

    if faulty:
        print(f'faulty_prompts: {faulty_prompts}')
        print(f'faulty_prompts_norm: {faulty_prompts_norm}')
