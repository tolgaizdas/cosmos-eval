from datasets import load_dataset

from tasks.task import Task
from tasks.xcopa.utils import process_doc


class XCOPA(Task):
    def __init__(self, n_shots=0):
        super().__init__('xcopa', n_shots=n_shots, prompt_intro="Öncül", prompt_conclusion="Cevap")

    def get_datasets(self):
        xcopa_ds = load_dataset('xcopa', 'tr')

        xcopa_valid = xcopa_ds['test']
        xcopa_valid = xcopa_valid.map(process_doc, load_from_cache_file=False)

        return xcopa_valid, xcopa_valid

    def get_attributes(self, data):
        ctx = data["ctx"]
        choices = data["choices"]
        gold = data["label"]
        gold_text = choices[gold]

        return ctx, choices, gold, gold_text
