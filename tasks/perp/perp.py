import pandas as pd
from datasets import Dataset

from tasks.task import Task


class Perp(Task):
    def __init__(self, n_shots=0):
        super().__init__('perp', n_shots=n_shots)
        self.train_ds, self.valid_ds = self.get_datasets()

    def get_datasets(self, ds_name="medium_long_tr.csv"):
        df = pd.read_csv(f"tasks/perp/ds/{ds_name}")
        ds = Dataset.from_pandas(df)
        return None, ds

    def get_attributes(self, data):
        context = data["Text"]
        return context, None, None, None  # Perplexity task only requires the context
