from datasets import load_dataset

from tasks.race.utils import process_doc
from tasks.task import Task
from utils.ds_utils import translate


class Race(Task):
    def __init__(self, n_shots=0):  # TODO: Check for default n_shots
        super().__init__('race', n_shots=n_shots, prompt_intro="", prompt_conclusion="")

    def get_datasets(self):
        race_ds = load_dataset("ehovy/race", "all")
        race_train = race_ds["train"]
        race_valid = race_ds["validation"]

        race_train = race_train.map(process_doc, load_from_cache_file=False)
        race_valid = race_valid.map(process_doc, load_from_cache_file=False)

        return race_train, race_valid

    def get_attributes(self, data):
        ctx = data["ctx"]
        choices = data["choices"]
        gold = ord(data["answer"]) - 65  # A: 0, B: 1, C: 2, D: 3
        gold_text = choices[gold]

        # TODO: Use translated dataset instead of translating each time
        ctx = translate(ctx)
        choices = [translate(choice) for choice in choices]
        gold_text = translate(gold_text)

        return ctx, choices, gold, gold_text
