from deep_translator import GoogleTranslator


def google_translate(text, source="auto", target="tr"):
    translator = GoogleTranslator(source=source, target=target)
    result = translator.translate(text)
    return result


def translate(text, source="auto", target="tr"):
    try:
        if len(text) < 5000:
            translated = google_translate(text, source=source, target=target)
        else:
            batches = [text[i:i + 4999] for i in range(0, len(text), 4999)]  # Split text into batches of 4999 characters (limit is 5000)
            translated_batches = [google_translate(batch, source=source, target=target) for batch in batches]
            translated = " ".join(translated_batches)
        return translated
    except Exception:
        return ""  # TODO: Returning an empty string causes errors on the encoding step.


def limit_dataset(ds, limit):
    if limit is None:
        return ds

    if limit > ds.num_rows:
        print(f"Limit is greater than the number of samples in the dataset. Setting limit to {ds.num_rows}.")
        limit = ds.num_rows

    return ds.select(range(limit))
