def process_doc(doc):
    story = f"{doc['input_sentence_1_tr']} {doc['input_sentence_2_tr']} {doc['input_sentence_3_tr']} {doc['input_sentence_4_tr']}"
    out_doc = {"story_tr": story,
               "endings_tr": [doc['sentence_quiz1_tr'], doc['sentence_quiz2_tr']],
               "answer": int(doc["answer_right_ending"]), }
    return out_doc
