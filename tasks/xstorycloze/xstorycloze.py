from datasets import load_from_disk

from tasks.task import Task
from tasks.xstorycloze.utils import process_doc


class XStoryCloze(Task):
    def __init__(self, n_shots=0):  # TODO: Check for default n_shots
        super().__init__('xstorycloze', n_shots=n_shots, prompt_intro="Hikaye")

    def get_datasets(self):
        xstorycloze_train = load_from_disk("tasks/xstorycloze/ds/xstorycloze_train_tr")
        xstorycloze_valid = load_from_disk("tasks/xstorycloze/ds/xstorycloze_valid_tr")

        xstorycloze_train = xstorycloze_train.map(process_doc, load_from_cache_file=False)
        xstorycloze_valid = xstorycloze_valid.map(process_doc, load_from_cache_file=False)

        return xstorycloze_train, xstorycloze_valid

    def get_attributes(self, data):
        story = data["story_tr"]
        endings = data["endings_tr"]
        gold = int(data["answer"]) - 1  # 0, 1
        gold_text = endings[gold]

        return story, endings, gold, gold_text
