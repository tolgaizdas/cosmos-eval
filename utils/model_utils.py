def get_max_length(model):
    # GPT
    if model.config.model_type == 'gpt2':
        return model.config.n_positions

    # BERT and LLAMA
    return model.config.max_position_embeddings