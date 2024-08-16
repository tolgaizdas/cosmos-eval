from datasets import load_dataset

from tasks.task import Task


class Perp(Task):
    def __init__(self, n_shots=0):
        super().__init__('perp', n_shots=n_shots)
        self.train_ds, self.valid_ds = self.get_datasets()

    def get_datasets(self):
        ds = load_dataset("malhajar/arc-tr-v0.2", split="test")  # TODO: Load the perplexity dataset instead
        return None, ds

    def get_attributes(self, data):
        context = data["question"]  # TODO: Change this to the context attribute
        return context, None, None, None  # Perplexity task only requires the context
