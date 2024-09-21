import numpy as np

def get_metrics(args, task_name):
    metrics = ["acc", "acc_norm", "perplexity"]
    if args.exclude_acc or task_name == 'perp':
        metrics.remove("acc")
    if args.exclude_acc_norm or task_name == 'perp':
        metrics.remove("acc_norm")
    if task_name == 'perp' and (args.exclude_acc or args.exclude_acc_norm):
        print('Perplexity task does not require acc and acc_norm. Excluding them.')
    if args.exclude_perplexity:
        metrics.remove("perplexity")
    return metrics


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
