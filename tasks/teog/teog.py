import pandas as pd
from datasets import Dataset

from tasks.task import Task


class TEOG(Task):
    def __init__(self):
        super().__init__('teog')
        self.train_ds, self.valid_ds = self.get_datasets()
        self.prompt_initial = "Soru"

    def get_datasets(self):
        teog = pd.read_csv("https://huggingface.co/datasets/aliardaf/LLMs-Turkish-TEOG-Leaderboard/resolve/main/teog_2013_text.csv")
        ds = Dataset.from_pandas(teog)
        train_ds, valid_ds = None, ds
        return train_ds, valid_ds

    def get_attributes(self, data):
        question = data["soru"]
        options = [data["cevapa"], data["cevapb"], data["cevapc"], data["cevapd"]]
        gold = ord(data["dogrucevap"]) - 65  # A: 0, B: 1, C: 2, D: 3
        gold_text = options[gold]

        return question, options, gold, gold_text
