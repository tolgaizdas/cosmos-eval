import math
import random
from abc import ABC, abstractmethod

from tqdm import tqdm

from utils.ds_utils import limit_dataset
from utils.eval_utils import get_results, perplexity, calculate_metrics


class Task(ABC):
    def __init__(self, name, n_shots=0, prompt_intro="İçerik", prompt_conclusion="Cevap"):
        self.name = name

        self.train_ds, self.valid_ds = self.get_datasets()

        self.n_shots = n_shots  # Number of shots

        self.prompt_intro = prompt_intro  # Start prompt word for the task
        self.prompt_conclusion = prompt_conclusion  # End prompt word for the task

        sample_choices = self.get_attributes(self.valid_ds[0])[1]
        self.expected_acc = 1.0 / len(sample_choices) if sample_choices else None

    def generate_prompt(self, ctx, ctx_choices, include_choices=False):
        prompt = ""

        if self.train_ds is None and self.n_shots > 0:
            print("Training dataset is not available. Setting n_shots to 0.")
            self.n_shots = 0

        if self.train_ds is not None and self.n_shots > 0:
            seed = random.randint(0, 1000)
            few_shots = self.train_ds.shuffle(seed=seed)
            n = 0  # Number of shots
            for shot in few_shots:
                if n == self.n_shots:
                    break

                try:
                    context, choices, _, gold_text = self.get_attributes(shot)
                except Exception:
                    continue

                if context == ctx:
                    continue

                prompt += (f"{self.prompt_intro}: " if self.prompt_intro else '') + context + "\n"
                if include_choices:
                    for j in range(len(choices)):
                        prompt += f"{chr(j + 65)}. " + choices[j] + "\n"

                prompt += (f"{self.prompt_conclusion}: " if self.prompt_conclusion else '') + gold_text + "\n\n"
                n += 1

        prompt += (f"{self.prompt_intro}: " if self.prompt_intro else '') + ctx + "\n"
        if include_choices:
            for j in range(len(ctx_choices)):
                prompt += f"{chr(j + 65)}. " + ctx_choices[j] + "\n"
        prompt += f"{self.prompt_conclusion}: " if self.prompt_conclusion else ''
        return prompt

    def eval_task(self, model, tokenizer, device, metrics, limit=None, faulty=False, include_choices=False):
        model.to(device)
        model.eval()

        metric_args = {metric: [] for metric in metrics}

        faulty_prompts = [] if faulty else None
        faulty_prompts_norm = [] if faulty else None

        ds = limit_dataset(self.valid_ds, limit)
        for data in tqdm(ds, desc="Evaluating"):
            try:
                context, choices, gold, _ = self.get_attributes(data)
            except Exception:
                continue

            if "acc" in metrics or "acc_norm" in metrics:
                prompt = self.generate_prompt(context, choices, include_choices)
                results, results_norm = get_results(model, tokenizer, prompt, choices, device)

                # Accuracy
                if "acc" in metrics:
                    predicted_index = results.index(max(results))
                    if faulty and predicted_index != gold:
                        faulty_prompts.append(prompt)
                    metric_args["acc"].append(1.0 if predicted_index == gold else 0.0)

                # Normalized accuracy
                if "acc_norm" in metrics:
                    predicted_index_norm = results_norm.index(max(results_norm))
                    if faulty and predicted_index_norm != gold:
                        faulty_prompts_norm.append(prompt)
                    metric_args["acc_norm"].append(1.0 if predicted_index_norm == gold else 0.0)

            # Perplexity
            if "perplexity" in metrics:
                perp = perplexity(model, tokenizer, context, device)
                if not math.isnan(perp):
                    metric_args["perplexity"].append(perp)

        ret = calculate_metrics(metric_args, expected_acc=self.expected_acc)
        return ret, faulty_prompts, faulty_prompts_norm

    @abstractmethod
    def get_attributes(self, data):
        pass

    @abstractmethod
    def get_datasets(self):
        pass
