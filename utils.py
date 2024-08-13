import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_byte_length(tokenizer, token_id):
    token_string = tokenizer.decode(token_id)
    token_bytes = token_string.encode('utf-8')
    byte_length = len(token_bytes)
    return byte_length


def get_log_probs(model, encoded_text):
    with torch.no_grad():
        outputs = model(encoded_text, labels=encoded_text)
        logits = outputs.logits
    logits = logits[0, -1, :]  # Get the logits of the last token
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.log(probs)
    return log_probs


def get_results(model, tokenizer, prompt, choices, device):
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    results, results_norm = [], []
    for choice in choices:
        unnormalized, normalized = 0.0, 0.0
        byte_length = 0
        current_prompt_ids = prompt_ids.clone()
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)

        for c_id in choice_ids:
            log_probs = get_log_probs(model, current_prompt_ids)
            unnormalized += log_probs[c_id].item()  # Unnormalized (https://blog.eleuther.ai/multiple-choice-normalization/)
            current_prompt_ids = torch.cat([current_prompt_ids, torch.tensor([c_id], device=device).unsqueeze(0)], dim=1)
            byte_length += get_byte_length(tokenizer, c_id)

        normalized += unnormalized / byte_length if byte_length > 0 else 0.0  # Byte-length normalized (https://blog.eleuther.ai/multiple-choice-normalization/)
        results.append(unnormalized)
        results_norm.append(normalized)
    return results, results_norm


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
        task = Hellaswag(n_shots) if n_shots is not None else Hellaswag()  # TODO: default n_shots can be handled in the task class
    elif task_name == 'arc':
        from tasks.arc.arc import ARC
        task = ARC(n_shots) if n_shots is not None else ARC()
    elif task_name == 'teog':
        from tasks.teog.teog import TEOG
        task = TEOG(n_shots) if n_shots is not None else TEOG()
    else:
        raise ValueError(f'Unknown task: {task_name}')

    if n_shots is None:
        print(f'n_shots is not provided. Using default value: {task.n_shots}')

    return task
