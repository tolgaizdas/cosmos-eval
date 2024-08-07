from abc import ABC, abstractmethod

from tqdm import tqdm

from utils import get_results


class Task(ABC):
    def __init__(self, name):
        self.name = name

        self.train_ds = None  # Training dataset
        self.valid_ds = None  # Validation dataset

        self.prompt_initial = None  # Initial prompt word for the task

    def generate_prompt(self, ctx, n_shots, include_choices=False):
        prompt = ""

        if self.train_ds is None and n_shots > 0:
            print("Training dataset is not available. Setting n_shots to 0.")
            n_shots = 0

        if self.train_ds is not None and n_shots > 0:
            random_data = self.train_ds.shuffle(seed=42).select(range(n_shots))
            for i, data in enumerate(random_data):
                context, choices, _, gold_text = self.get_attributes(data)

                prompt += f"{self.prompt_initial}: {context}\n"
                if include_choices:
                    for j in range(len(choices)):
                        prompt += f"{chr(j + 65)}. {choices[j]}\n"

                prompt += f"Cevap: {gold_text}\n\n"

        prompt += f"{self.prompt_initial}: {ctx}\n"
        prompt += "Cevap:"
        return prompt

    def eval_task(self, model, tokenizer, n_shots, device):
        correct_norm, total_norm = 0.0, 0.0
        correct, total = 0.0, 0.0

        model.to(device)
        model.eval()

        for data in tqdm(self.valid_ds, desc="Evaluating"):
            try:
                context, choices, gold, _ = self.get_attributes(data)
            except Exception:
                continue

            prompt = self.generate_prompt(context, n_shots)
            results, results_norm = get_results(model, tokenizer, prompt, choices, device)

            # Accuracy
            predicted_index = results.index(max(results))
            correct += 1.0 if predicted_index == gold else 0.0
            total += 1.0

            # Normalized accuracy
            predicted_index_norm = results_norm.index(max(results_norm))
            correct_norm += 1.0 if predicted_index_norm == gold else 0.0
            total_norm += 1.0

        acc = correct / total if total > 0 else 0.0
        acc_norm = correct_norm / total_norm if total_norm > 0 else 0.0

        return acc, acc_norm

    @abstractmethod
    def get_attributes(self, data):
        pass

    @abstractmethod
    def get_datasets(self):
        pass
