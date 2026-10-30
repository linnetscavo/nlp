import os
import re

os.environ["HF_HOME"] = r"F:\huggingface"
os.environ["HF_HUB_CACHE"] = r"F:\huggingface\hub"
os.environ["HF_XET_CACHE"] = r"F:\huggingface\xet"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig


MODEL_NAME = "ai-forever/rugpt3large_based_on_gpt2"
TARGET_WORDS = ["психологическое", "опосредованным"]


PROMPT = (
    "Сравним воздействие на тело без посредника и влияние на сознание через посредника.\n"
    "Физика — физическое воздействие, которое является непосредственным.\n"
    "Психология —"
)


def check_prompt(text):
    words = re.findall(r"\w+", text.lower())
    if any(word.startswith(("психологическ", "опосредован")) for word in words):
        raise ValueError("Промпт содержит искомое слово или его форму.")


def check_result(text):
    words = re.findall(r"\w+", text.lower())
    first_seen = False
    for word in words:
        if word == TARGET_WORDS[0]:
            first_seen = True
        elif word == TARGET_WORDS[1] and first_seen:
            return True
    return False


def generate(model, tok, text):
    inputs = tok(text, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    max_new_tokens = 64
    if prompt_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Промпт и продолжение превышают размер контекста модели.")

    config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            generation_config=config,
        )

    return tok.decode(out[0, prompt_length:], skip_special_tokens=True)


def main():
    check_prompt(PROMPT)

    device = "cpu"
    torch.set_num_threads(6)
    print(f"Загрузка модели. Устройство: {device}", flush=True)
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    model.eval()

    continuation = generate(model, tok, PROMPT)
    print(f"\nПромпт:\n{PROMPT}")
    print(f"\nПродолжение модели:\n{continuation}")
    print(f"\nУсловие выполнено: {check_result(continuation)}")


if __name__ == "__main__":
    main()
