from abc import ABC, abstractmethod

import math
from tqdm import tqdm

from utils import get_results, perplexity

import logging
logging.getLogger("datasets").setLevel(logging.WARNING)  # Suppressing Hugging Face datasets INFO and DEBUG messages

class Task(ABC):
    def __init__(self, name, n_shots=0, prompt_initial="Context"):
        self.name = name

        self.train_ds = None  # Training dataset
        self.valid_ds = None  # Validation dataset

        self.n_shots = n_shots  # Number of shots

        self.prompt_initial = prompt_initial  # Initial prompt word for the task

    def generate_prompt(self, ctx, idx, include_choices=False):
        prompt = ""

        if self.train_ds is None and self.n_shots > 0:
            print("Training dataset is not available. Setting n_shots to 0.")
            self.n_shots = 0

        if self.train_ds is not None and self.n_shots > 0:
            # Filter out the context to avoid repetition if the validation and training datasets are the same
            # Assuming there is no overlap between the validation and training datasets when they are different
            filtered_ds = self.train_ds.filter(lambda data: data['idx'] != idx)
            few_shots = filtered_ds.shuffle(seed=42).select(range(self.n_shots))
            for shot in few_shots:
                context, choices, _, gold_text = self.get_attributes(shot)

                prompt += f"{self.prompt_initial}: {context}\n"
                if include_choices:
                    for j in range(len(choices)):
                        prompt += f"{chr(j + 65)}. {choices[j]}\n"

                prompt += f"Cevap: {gold_text}\n\n"

        prompt += f"{self.prompt_initial}: {ctx}\n"
        prompt += "Cevap:"
        return prompt

    def eval_task(self, model, tokenizer, device, metrics, limit, faulty):
        model.to(device)
        model.eval()

        ret = {metric: 0.0 for metric in metrics}
        total_samples = 0

        faulty_prompts = [] if faulty else None
        faulty_prompts_norm = [] if faulty else None

        self.train_ds, self.valid_ds = self.process_datasets(limit)
        for data in tqdm(self.valid_ds, desc="Evaluating"):
            try:
                context, choices, gold, _ = self.get_attributes(data)
            except Exception:
                continue

            if "acc" in metrics or "acc_norm" in metrics:
                prompt = self.generate_prompt(context, data['idx'])
                results, results_norm = get_results(model, tokenizer, prompt, choices, device)

                # Accuracy
                if "acc" in metrics:
                    predicted_index = results.index(max(results))
                    if faulty and predicted_index != gold:
                        faulty_prompts.append(prompt)
                    ret["acc"] += 1.0 if predicted_index == gold else 0.0

                # Normalized accuracy
                if "acc_norm" in metrics:
                    predicted_index_norm = results_norm.index(max(results_norm))
                    if faulty and predicted_index_norm != gold:
                        faulty_prompts_norm.append(prompt)
                    ret["acc_norm"] += 1.0 if predicted_index_norm == gold else 0.0

            # Perplexity
            if "perplexity" in metrics:
                perp = perplexity(model, tokenizer, context, device)
                if not math.isnan(perp):
                    ret["perplexity"] += perp

            total_samples += 1

        if total_samples > 0:
            for metric in metrics:
                ret[metric] /= total_samples

        return ret, faulty_prompts, faulty_prompts_norm

    def process_datasets(self, limit):
        if limit is not None and limit > self.valid_ds.num_rows:
            print(f"Limit is greater than the number of samples in the dataset. Setting limit to {self.valid_ds.num_rows}.")
            limit = self.valid_ds.num_rows
        # Limit the number of samples in the validation dataset
        # No need to limit the training dataset since it is used for generating prompts
        valid_ds = self.valid_ds.select(range(limit))

        def add_index(d, idx):
            data = d.copy()
            data['idx'] = idx
            return data

        # Add index to the dataset
        train_ds = self.train_ds.map(lambda data, idx: add_index(data, idx), with_indices=True) if self.train_ds is not None else None
        valid_ds = valid_ds.map(lambda data, idx: add_index(data, idx), with_indices=True)

        return train_ds, valid_ds

    @abstractmethod
    def get_attributes(self, data):
        pass

    @abstractmethod
    def get_datasets(self):
        pass
