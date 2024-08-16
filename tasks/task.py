from abc import ABC, abstractmethod

from tqdm import tqdm

from utils import get_results, perplexity


class Task(ABC):
    def __init__(self, name, n_shots=0, prompt_initial="Context"):
        self.name = name

        self.train_ds = None  # Training dataset
        self.valid_ds = None  # Validation dataset

        self.n_shots = n_shots  # Number of shots

        self.prompt_initial = prompt_initial  # Initial prompt word for the task

    def generate_prompt(self, ctx, include_choices=False):
        prompt = ""

        if self.train_ds is None and self.n_shots > 0:
            print("Training dataset is not available. Setting n_shots to 0.")
            self.n_shots = 0

        if self.train_ds is not None and self.n_shots > 0:
            random_data = self.train_ds.shuffle(seed=42).select(range(self.n_shots))  # TODO: Random data should not include the context (ctx)
            for data in random_data:
                context, choices, _, gold_text = self.get_attributes(data)

                prompt += f"{self.prompt_initial}: {context}\n"
                if include_choices:
                    for j in range(len(choices)):
                        prompt += f"{chr(j + 65)}. {choices[j]}\n"

                prompt += f"Cevap: {gold_text}\n\n"

        prompt += f"{self.prompt_initial}: {ctx}\n"
        prompt += "Cevap:"
        return prompt

    def eval_task(self, model, tokenizer, device, limit, faulty):
        model.to(device)
        model.eval()

        correct, total = 0.0, 0.0
        correct_norm, total_norm = 0.0, 0.0
        
        faulty_prompts = [] if faulty else None
        faulty_prompts_norm = [] if faulty else None

        ds = self.limit_dataset(limit)
        for data in tqdm(ds, desc="Evaluating"):
            try:
                context, choices, gold, _ = self.get_attributes(data)
            except Exception:
                continue

            prompt = self.generate_prompt(context)
            results, results_norm = get_results(model, tokenizer, prompt, choices, device)

            # Accuracy
            predicted_index = results.index(max(results))
            if faulty and predicted_index != gold:
                faulty_prompts.append(prompt)
            correct += 1.0 if predicted_index == gold else 0.0
            total += 1.0

            # Normalized accuracy
            predicted_index_norm = results_norm.index(max(results_norm))
            if faulty and predicted_index_norm != gold:
                faulty_prompts_norm.append(prompt)
            correct_norm += 1.0 if predicted_index_norm == gold else 0.0
            total_norm += 1.0

        acc = correct / total if total > 0 else 0.0
        acc_norm = correct_norm / total_norm if total_norm > 0 else 0.0

        return acc, acc_norm, faulty_prompts, faulty_prompts_norm

    def eval_perplexity(self, model, tokenizer, device, limit):
        model.to(device)
        model.eval()

        avg = 0.0
        samples = 0

        ds = self.limit_dataset(limit)
        for data in tqdm(ds, desc="Evaluating"):
            ctx, _, _, _ = self.get_attributes(data)
            avg += perplexity(model, tokenizer, ctx, device)
            samples += 1
        perplexity_score = avg / samples if samples > 0 else 0.0
        return perplexity_score.item()
    
    def limit_dataset(self, limit):
        if limit is None:
            return self.valid_ds

        if limit > self.valid_ds.num_rows:
            print(f"Limit is greater than the number of samples in the dataset. Setting limit to {self.valid_ds.num_rows}.")
            limit = self.valid_ds.num_rows
        
        ds = self.valid_ds.select(range(limit))
        return ds

    @abstractmethod
    def get_attributes(self, data):
        pass

    @abstractmethod
    def get_datasets(self):
        pass
