from datasets import load_dataset

from tasks.hellaswag.utils import process_doc
from tasks.task import Task


class Hellaswag(Task):
    def __init__(self, n_shots=10):
        super().__init__('hellaswag', n_shots=n_shots)
        self.train_ds, self.valid_ds = self.get_datasets()
        self.prompt_initial = "İçerik"

    def get_datasets(self):
        hellaswag = load_dataset("malhajar/hellaswag-tr")
        hellaswag_valid_v02 = load_dataset("malhajar/hellaswag_tr-v0.2", split="validation")

        hellaswag_train = hellaswag["train"]
        hellaswag_valid = hellaswag_valid_v02  # 8k
        # hellaswag_valid = concatenate_datasets([hellaswag_valid_v02, hellaswag["validation"]])  # 18k

        hellaswag_train = hellaswag_train.map(process_doc)
        hellaswag_valid = hellaswag_valid.map(process_doc)

        return hellaswag_train, hellaswag_valid

    def get_attributes(self, data):
        context = data["ctx"]
        endings = data["endings"]
        gold = data["label"]
        gold_text = endings[gold]
        return context, endings, gold, gold_text
