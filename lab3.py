import sys

import torch
from transformers import BertTokenizer, BertForMaskedLM

MODEL_NAME = "google-bert/bert-base-multilingual-uncased"
TARGET_WORDS = ["стало", "рано"]

TEXT = "Уже [MASK] светло, и дети вышли во двор."


def predict_words(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise ValueError("Предложение слишком длинное для модели.")

    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

    with torch.no_grad():
        output = model(**inputs)

    probabilities = torch.softmax(output.logits[0, mask_index], dim=-1)
    top = torch.topk(probabilities, 10)
    words = [tokenizer.decode([token_id]) for token_id in top.indices.tolist()]
    return words, top.values.tolist()


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    text = sys.argv[1] if len(sys.argv) > 1 else TEXT
    words, probabilities = predict_words(text, tokenizer, model)

    print(f"Предложение: {text}\n")
    for rank, (word, probability) in enumerate(zip(words, probabilities), start=1):
        print(f"{rank:2}. {word:20} {probability:.4%}")

    print()
    for word in TARGET_WORDS:
        if word in words:
            print(f"{word}: место {words.index(word) + 1}")
        else:
            print(f"{word}: нет в первой десятке")

    if all(word in words for word in TARGET_WORDS):
        print("Условие выполнено: оба слова входят в топ-10 в нужной словоформе.")
    else:
        print("Условие не выполнено: нужно изменить предложение.")


if __name__ == "__main__":
    main()
