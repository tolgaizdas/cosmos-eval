import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def relative_accuracy(acc, expected_acc):
    if expected_acc is None:
        return acc

    if acc < expected_acc:
        return 0.0

    return (acc - expected_acc) / (1.0 - expected_acc)


def calculate_metrics(metric_args, expected_acc=None, decimals=4):
    ret = {}
    for metric, value in metric_args.items():
        ret[metric] = np.mean(value).item() if value else float('nan')

    if "acc" in ret:
        ret["rel_acc"] = relative_accuracy(ret["acc"], expected_acc) if ret["acc"] else 0.0

    if "acc_norm" in ret:
        ret["rel_acc_norm"] = relative_accuracy(ret["acc_norm"], expected_acc) if ret["acc_norm"] else 0.0

    ret = {metric: round(value, decimals) for metric, value in ret.items()}  # Round to 4 decimal places
    return ret


def generate_prompt(context, choices, gold_text=None, intro=None, conclusion=None, include_choices=False, previous_tokens=False):
    intro = f"{intro}: " if intro else ''
    conclusion = f"{conclusion}: " if conclusion else ''

    if previous_tokens:
        model = AutoModelForCausalLM.from_pretrained("path/to/model")  # TODO: Change this
        tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")  # TODO: Change this
        reversed_context = tokenizer.decode(tokenizer.encode(context)[::-1], skip_special_tokens=True)
        text_generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=0,
            eos_token_id=tokenizer.eos_token_id # Use EOS token to stop early
        )
        r = text_generator(reversed_context, max_length=max(100, len(reversed_context)), truncation=True)[0]['generated_text']
        context_with_prev = tokenizer.decode(tokenizer.encode(r)[::-1])
        prompt = [intro + context_with_prev + "\n"]
    else:
        prompt = [intro + context + "\n"]

    if include_choices:
        for j in range(len(choices)):
            prompt.append(f"{chr(j + 65)}. " + choices[j] + "\n")

    prompt.append(conclusion)

    if gold_text:
        prompt.append(gold_text + "\n\n")

    return ''.join(prompt)
