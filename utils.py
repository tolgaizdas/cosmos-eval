import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_encoded_input(text, tokenizer, device):
    return tokenizer.encode(text, return_tensors="pt").to(device)


def get_score(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor, labels=input_tensor)
        loss = outputs.loss.item()
    return -loss  # Lower loss better score


def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_task(task_name):
    task_name = task_name.lower().strip()

    if task_name == 'hellaswag':
        from tasks.hellaswag.hellaswag import Hellaswag
        return Hellaswag()
    elif task_name == 'arc':
        from tasks.arc.arc import ARC
        return ARC()
    else:
        raise ValueError(f'Unknown task: {task_name}')
