from tqdm import tqdm

from utils import get_encoded_input, get_score


class Task:
    def __init__(self, name):
        self.name = name

        self.train_ds = None
        self.valid_ds = None

        self.prompt_initial = None  # Initial prompt word for the task
        self.label_map = None  # Mapping of choice index to label

    def generate_prompt(self, ctx, n_shots, include_choices=False):
        prompt = ""
        random_data = self.train_ds.shuffle(seed=42).select(range(n_shots))

        for i, data in enumerate(random_data):
            context, choices, _, gold_text = self.get_attributes(data)

            prompt += f"{self.prompt_initial}: {context}\n"
            if include_choices:
                for j in range(len(choices)):
                    label = self.label_map[j] if self.label_map is not None else chr(j + 65)
                    prompt += f"{label}. {choices[j]}\n"

            prompt += f"Cevap: {gold_text}\n\n"

        prompt += f"{self.prompt_initial}: {ctx}\n"
        prompt += "Cevap:"
        return prompt

    def eval_task(self, model, tokenizer, n_shots, device):
        correct_norm, total_norm = 0, 0
        correct, total = 0, 0

        model.to(device)
        model.eval()

        for data in tqdm(self.valid_ds, desc="Evaluating"):
            context, choices, gold, _ = self.get_attributes(data)

            prompt = self.generate_prompt(context, n_shots)
            encoded_inputs = [get_encoded_input(f"{prompt} {choice}", tokenizer, device) for choice in choices]

            # Accuracy
            scores = [get_score(model, input_tensor) for input_tensor in encoded_inputs]
            predicted_index = scores.index(max(scores))
            correct += int(predicted_index == gold)
            total += 1

            # Normalized accuracy
            scores_norm = [score / len(choice) if len(choice) > 0 else 0 for score, choice in zip(scores, choices)]
            predicted_index_norm = scores_norm.index(max(scores_norm))
            correct_norm += int(predicted_index_norm == gold)
            total_norm += 1

        acc = correct / total if total > 0 else 0
        acc_norm = correct_norm / total_norm if total_norm > 0 else 0

        return acc, acc_norm

    def get_attributes(self, data):
        raise NotImplementedError

    def get_datasets(self):
        raise NotImplementedError
