from datasets import load_from_disk

from tasks.task import Task


class OpenBookQA(Task):
    def __init__(self, n_shots=0):  # TODO: Check for default n_shots
        super().__init__('openbookqa', n_shots=n_shots)
        self.train_ds, self.valid_ds = self.get_datasets()
        self.prompt_intro = "Soru"

    def get_datasets(self):
        openbookqa_train = load_from_disk("tasks/openbookqa/ds/openbookqa_train_tr")
        openbookqa_valid = load_from_disk("tasks/openbookqa/ds/openbookqa_valid_tr")

        return openbookqa_train, openbookqa_valid

    def get_attributes(self, data):
        question = data["question_stem_tr"]
        choices = data["choices_tr"]
        gold = ord(data["answerKey"]) - 65  # A: 0, B: 1, C: 2, D: 3
        gold_text = choices[gold]

        return question, choices, gold, gold_text
