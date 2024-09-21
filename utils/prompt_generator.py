from transformers import pipeline

from utils.base_utils import load_model_and_tokenizer


class PromptGenerator:
    def __init__(self, prompt_intro, prompt_conclusion):
        self.intro = prompt_intro
        self.conclusion = prompt_conclusion

        self.model, self.tokenizer = None, None  # Model and tokenizer for generating previous tokens

    def generate_previous_tokens(self, context, device="cpu"):
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_model_and_tokenizer("path/to/model", device)

        model, tokenizer = self.model, self.tokenizer  # Just to make it easier to read

        reversed_context = tokenizer.decode(tokenizer.encode(context)[::-1], skip_special_tokens=True)
        text_generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            eos_token_id=tokenizer.eos_token_id  # Use EOS token to stop early
        )
        r = text_generator(reversed_context, max_length=max(100, len(reversed_context)), truncation=True)[0]['generated_text']
        context_with_prev = tokenizer.decode(tokenizer.encode(r)[::-1])
        return context_with_prev

    def generate_prompt(self, context, choices, gold_text=None, include_choices=False, previous_tokens=False, device="cpu"):
        intro = f"{self.intro}: " if self.intro else ''
        conclusion = f"{self.conclusion}: " if self.conclusion else ''

        if previous_tokens:
            context = self.generate_previous_tokens(context, device)

        prompt = [intro + context + "\n"]

        if include_choices:
            for j in range(len(choices)):
                prompt.append(f"{chr(j + 65)}. " + choices[j] + "\n")

        prompt.append(conclusion)

        if gold_text:
            prompt.append(gold_text + "\n\n")

        return ''.join(prompt)
