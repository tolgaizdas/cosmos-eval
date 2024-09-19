def fill_in_the_blank(question, option):
    question = question.replace("_", option)  # Replace the blank directly with the option
    question = " ".join(question.split())  # Remove any extra spaces between words
    question = question.strip()
    return question


def filter_masked_question(data):
    return "_" not in data["question"]


def filter_masked_article(data):
    return "_" not in data["article"]


def filter_doc(doc):
    return filter_masked_article(doc) and filter_masked_question(doc)


def process_doc(doc):
    """
    article = doc["article"]
    question = doc["question"]
    options = doc["options"]
    choices = []
    for option in options:
        if "_" in question:
            choices.append(fill_in_the_blank(question, option))
        else:
            choices.append(option)

    ctx = f'{article}\n\n{question}' if "_" not in question else f'{article}\n\nBuna göre;'  # OLD_TODO: Find a better way to handle this
    out_doc = {"ctx": ctx, "choices": choices}
    """
    article = doc["article_tr"]
    article = article.replace("\n", " ")
    article = " ".join(article.split())  # Remove any extra spaces between words
    ctx = f'{article}\nSoru: {doc["question_tr"]}'
    out_doc = {"ctx_tr": ctx}
    return out_doc
