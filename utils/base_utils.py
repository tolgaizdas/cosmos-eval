from transformers import AutoModelForCausalLM, AutoTokenizer


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
    else:
        raise ValueError(f'Unknown task: {task_name}')

    if n_shots is None:
        print(f'n_shots is not provided. Using default value: {task.n_shots}')

    return task
