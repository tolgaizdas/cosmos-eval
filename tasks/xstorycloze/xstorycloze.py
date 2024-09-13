from datasets import load_dataset

from tasks.task import Task
from tasks.xstorycloze.utils import process_doc
from utils.ds_utils import translate


class XStoryCloze(Task):
    def __init__(self, n_shots=0):  # TODO: Check for default n_shots
        super().__init__('xstorycloze', n_shots=n_shots)
        self.train_ds, self.valid_ds = self.get_datasets()
        self.prompt_intro = "Hikaye"

    def get_datasets(self):
        xstorycloze_ds = load_dataset("juletxara/xstory_cloze", "en")
        xstorycloze_train = xstorycloze_ds["train"]
        xstorycloze_valid = xstorycloze_ds["eval"]

        xstorycloze_train = xstorycloze_train.map(process_doc, load_from_cache_file=False)
        xstorycloze_valid = xstorycloze_valid.map(process_doc, load_from_cache_file=False)

        return xstorycloze_train, xstorycloze_valid

    def get_attributes(self, data):
        story = data["story"]
        endings = data["endings"]
        gold = data["answer"]  # 1, 2
        gold_text = endings[gold - 1]  # -1 to convert (1, 2) to (0, 1)

        # TODO: Use translated dataset instead of translating each time
        story = translate(story)
        endings = [translate(ending) for ending in endings]
        gold_text = translate(gold_text)

        return story, endings, gold, gold_text
