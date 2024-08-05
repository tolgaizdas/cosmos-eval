import re


def preprocess(text):
    text = text.replace(" [title]", ". ")
    text = text.replace(" [header]", ". ")
    text = text.replace(" [başlık]", ". ")
    text = text.replace(" [adım]", ". ")
    text = text.replace("  ", " ")
    text = text.replace("..", ".")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.strip()
    return text


def process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    out_doc = {"ctx": preprocess(ctx), "endings": [preprocess(ending) for ending in doc["endings"]],
               "label": int(doc["label"]), }
    return out_doc
