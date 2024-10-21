def convert_choice(choice):
    return choice[0].lower() + choice[1:]


def process_doc(doc):
    choice1 = convert_choice(doc["choice1"])
    choice2 = convert_choice(doc["choice2"])
    question_tr = tr[doc["question"]]
    ctx = doc["premise"].strip()[:-1] + f" {question_tr}"
    out_doc = {"ctx": ctx, "choices": [choice1, choice2]}
    return out_doc


tr = {
    "cause": "çünkü",
    "effect": "bu yüzden"
}