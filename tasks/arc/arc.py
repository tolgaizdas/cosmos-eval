import datasets
from datasets import load_dataset

from tasks.task import Task


class ARC(Task):
    def __init__(self):
        super().__init__('arc')
        self.train_ds, self.valid_ds = self.get_datasets()
        self.prompt_initial = "Soru"
        self.label_map = ['A', 'B', 'C', 'D']

    def get_datasets(self):
        arc_ds = load_dataset("malhajar/arc-tr-v0.2")
        arc_train = arc_ds["test"]
        arc_valid = arc_ds["test"]
        return arc_train, arc_valid

    def get_attributes(self, data):
        question = data["question"]
        choices = data["choices"]["text"]
        gold = ord(data["answerKey"]) - 65  # A: 0, B: 1, C: 2, D: 3
        gold_text = choices[gold]
        return question, choices, gold, gold_text
