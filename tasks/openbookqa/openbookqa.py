from datasets import load_dataset

from tasks.task import Task
from utils.ds_utils import translate


class OpenBookQA(Task):
    def __init__(self, n_shots=0):  # TODO: Check for default n_shots
        super().__init__('openbookqa', n_shots=n_shots)
        self.train_ds, self.valid_ds = self.get_datasets()
        self.prompt_intro = "Soru"

    def get_datasets(self):
        openbookqa_ds = load_dataset("allenai/openbookqa")
        openbookqa_train = openbookqa_ds["train"]
        openbookqa_valid = openbookqa_ds["validation"]

        return openbookqa_train, openbookqa_valid

    def get_attributes(self, data):
        question = data["question_stem"]
        choices = data["choices"]["text"]
        gold = ord(data["answerKey"]) - 65  # A: 0, B: 1, C: 2, D: 3
        gold_text = choices[gold]

        # TODO: Use translated dataset instead of translating each time
        question = translate(question)
        choices = [translate(choice) for choice in choices]
        gold_text = translate(gold_text)

        return question, choices, gold, gold_text
