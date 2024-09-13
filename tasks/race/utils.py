def fill_in_the_blank(question, option):
    question = question.replace("_", option)  # Replace the blank directly with the option
    question = " ".join(question.split())  # Remove any extra spaces between words
    question = question.strip()
    return question


def process_doc(doc):
    article = doc["article"]
    question = doc["question"]
    options = doc["options"]
    choices = []

    for option in options:
        if "_" in question:
            choices.append(fill_in_the_blank(question, option))
        else:
            choices.append(option)

    ctx = f'{article}\n\n{question}' if "_" not in question else f'{article}\n\nBuna göre;'  # TODO: Find a better way to handle this
    out_doc = {"ctx": ctx, "choices": choices}
    return out_doc
