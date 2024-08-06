import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_encoded_input(text, tokenizer, device):
    return tokenizer.encode(text, return_tensors="pt").to(device)


def get_byte_length(tokenizer, token_id):
    # Decode the token ID to get the token string
    token_string = tokenizer.decode(token_id)

    # Encode the token string to bytes
    token_bytes = token_string.encode('utf-8')

    # Get the byte length
    byte_length = len(token_bytes)

    return byte_length


def get_results(model, tokenizer, prompt, choices, device):
    # Tokenize the prompt
    prompt_ids = get_encoded_input(prompt, tokenizer, device)

    # Get model output for the prompt
    with torch.no_grad():
        outputs = model(prompt_ids, labels=prompt_ids)
        logits = outputs.logits

    # Calculate probabilities
    logits = logits[0, -1, :]  # Get logits for the next token
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.log(probs)

    # Calculate sum of log probabilities for each choice
    results, results_norm = [], []
    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)
        unnormalized = sum(log_probs[c_id].item() for c_id in choice_ids)
        normalized = unnormalized / sum(get_byte_length(tokenizer, c_id) for c_id in choice_ids)
        results.append(unnormalized)
        results_norm.append(normalized)

    return results, results_norm


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
