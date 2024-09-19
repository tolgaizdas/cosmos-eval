from datasets import load_from_disk

from tasks.race.utils import process_doc
from tasks.task import Task


class Race(Task):
    def __init__(self, n_shots=0):  # TODO: Check for default n_shots
        super().__init__('race', n_shots=n_shots, prompt_intro="Metin", prompt_conclusion="Cevap")

    def get_datasets(self):
        #race_ds = load_dataset("ehovy/race", "all")
        # race_train = race_ds["train"]
        # race_train = race_train.map(process_doc, load_from_cache_file=False)
        # race_valid = race_ds["validation"]

        race_valid = load_from_disk("tasks/race/ds/race_valid_tr")
        race_valid = race_valid.map(process_doc, load_from_cache_file=False)

        return race_valid, race_valid

    def get_attributes(self, data):
        ctx = data["ctx_tr"]
        options = data["options_tr"]
        gold = ord(data["answer"]) - 65  # A: 0, B: 1, C: 2, D: 3
        gold_text = options[gold]

        return ctx, options, gold, gold_text
