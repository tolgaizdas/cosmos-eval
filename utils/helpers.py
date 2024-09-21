import numpy as np


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


def generate_prompt(context, choices, gold_text=None, intro=None, conclusion=None, include_choices=False):
    intro = f"{intro}: " if intro else ''
    conclusion = f"{conclusion}: " if conclusion else ''

    prompt = [intro + context + "\n"]

    if include_choices:
        for j in range(len(choices)):
            prompt.append(f"{chr(j + 65)}. " + choices[j] + "\n")

    prompt.append(conclusion)

    if gold_text:
        prompt.append(gold_text + "\n\n")

    return ''.join(prompt)
