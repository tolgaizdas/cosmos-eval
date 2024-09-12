import torch


def get_log_likelihood(model, input_ids, target_ids=None):
    if target_ids is None:
        target_ids = input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    return -neg_log_likelihood


def get_byte_length(tokenizer, token_id):
    token_string = tokenizer.decode(token_id)
    token_bytes = token_string.encode('utf-8')
    byte_length = len(token_bytes)
    return byte_length


def get_log_probs(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    logits = logits[0, -1, :]  # Get the logits of the last token
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.log(probs)
    return log_probs


def get_results(model, tokenizer, prompt, choices, device):
    prompt = prompt.strip() + " "  # Add space to separate prompt from choices
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    choice_ids_list = [tokenizer.encode(choice, return_tensors="pt").to(device) for choice in choices]
    all_ids = torch.cat([prompt_ids] + choice_ids_list, dim=1)

    attention_mask = torch.zeros(all_ids.shape, dtype=torch.long, device=device)
    attention_mask[:, :prompt_len] = 1

    choice_start_idx = prompt_len

    results, results_norm = [], []
    for choice_ids in choice_ids_list:
        unnormalized, normalized = 0.0, 0.0
        byte_length = 0

        choice_len = choice_ids.shape[1]
        choice_end_idx = choice_start_idx + choice_len
        for i in range(choice_len):
            attention_mask[:, choice_start_idx:choice_start_idx + i + 1] = 1
            log_probs = get_log_probs(model, all_ids, attention_mask)
            unnormalized += log_probs[choice_ids[0, i]].item()  # Un-normalized (https://blog.eleuther.ai/multiple-choice-normalization/)

        attention_mask[:, choice_start_idx:choice_end_idx] = 0
        choice_start_idx = choice_end_idx

        normalized += unnormalized / byte_length if byte_length > 0 else 0.0  # Byte-length normalized (https://blog.eleuther.ai/multiple-choice-normalization/)
        results.append(unnormalized)
        results_norm.append(normalized)

    return results, results_norm


def perplexity(model, tokenizer, text, device):
    """
     https://huggingface.co/docs/transformers/en/perplexity
     TODO: This function is taken from the Hugging Face documentation. It can be modified to be more efficient.
    """
    encodings = tokenizer(text, return_tensors="pt").to(device)

    max_length = model.config.n_positions
    stride = 512  # TODO: Stride can be an argument or a parameter of the model
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # May be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask the tokens before the target sequence length

        neg_log_likelihood = -1 * get_log_likelihood(model, input_ids, target_ids)
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
