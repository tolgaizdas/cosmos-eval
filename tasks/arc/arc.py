from datasets import load_dataset

from tasks.task import Task


class ARC(Task):
    def __init__(self, n_shots=25):
        super().__init__('arc', n_shots=n_shots, prompt_intro="Soru")

    def get_datasets(self):
        arc_ds = load_dataset("malhajar/arc-tr-v0.2")
        arc_valid = arc_ds["test"]
        return arc_valid, arc_valid

    def get_attributes(self, data):
        question = data["question"]
        choices = data["choices"]["text"]
        try:
            gold = int(data["answerKey"]) - 1  # 0, 1, 2, 3
        except ValueError:
            gold = ord(data["answerKey"]) - 65  # A: 0, B: 1, C: 2, D: 3
        gold_text = choices[gold]
        return question, choices, gold, gold_text
