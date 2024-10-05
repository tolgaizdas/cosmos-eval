import math
import random
from abc import ABC, abstractmethod

from tqdm import tqdm

from utils import PromptGenerator
from utils.ds_utils import limit_dataset
from utils.eval_utils import get_results, perplexity
from utils.metric_utils import calculate_metrics


class Task(ABC):
    def __init__(self, name, n_shots=0, prompt_intro="İçerik", prompt_conclusion="Cevap"):
        self.name = name

        self.train_ds, self.valid_ds = self.get_datasets()

        self.n_shots = n_shots  # Number of shots

        sample_choices = self.get_attributes(self.valid_ds[0])[1]
        self.expected_acc = 1.0 / len(sample_choices) if sample_choices else None

        self.prompt_generator = PromptGenerator(prompt_intro, prompt_conclusion)

    def get_prompt(self, data, include_choices=False, previous_tokens=False):
        def build_prompt(context, choices, gold_text=None):
            return self.prompt_generator.generate_prompt(context, choices, gold_text=gold_text, include_choices=include_choices, previous_tokens=previous_tokens)

        ctx, ctx_choices, _, _ = self.get_attributes(data)

        prompt = []

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

                prompt.append(build_prompt(context, choices, gold_text))
                n += 1

        prompt.append(build_prompt(ctx, ctx_choices))

        return "".join(prompt)

    def eval_task(self, model, tokenizer, metrics, limit=None, faulty=False, include_choices=False, previous_tokens=False):
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
                prompt = self.get_prompt(data, include_choices=include_choices, previous_tokens=previous_tokens)
                results, results_norm = get_results(model, tokenizer, prompt, choices, model.device)

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
                perp = perplexity(model, tokenizer, context, model.device)
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
