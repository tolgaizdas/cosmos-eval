def process_doc(doc):
    story = f"{doc['input_sentence_1']} {doc['input_sentence_2']} {doc['input_sentence_3']} {doc['input_sentence_4']}"
    out_doc = {"story": story,
               "endings": [doc['sentence_quiz1'], doc['sentence_quiz2']],
               "answer": int(doc["answer_right_ending"]), }
    return out_doc
