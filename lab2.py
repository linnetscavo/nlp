from pathlib import Path

import gensim


MODEL_PATH = Path(__file__).with_name("cbow.txt")
POSITIVE = ["развалина_NOUN", "будущее_NOUN"]
NEGATIVE = []
TARGETS = {"руины_NOUN", "перспектива_NOUN"}
TOP_N = 10


def nearest_nouns(model, positive, negative, count=10):
    limit = max(20, count)
    while True:
        neighbors = model.most_similar(
            positive=positive, negative=negative, topn=limit
        )
        nouns = [(word, score) for word, score in neighbors if word.endswith("_NOUN")]
        if len(nouns) >= count:
            return nouns[:count]
        if limit >= len(model):
            raise ValueError("Недостаточно существительных")
        limit = min(limit * 2, len(model))


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Не найдена модель: {MODEL_PATH}")

    model = gensim.models.KeyedVectors.load_word2vec_format(
        str(MODEL_PATH), binary=False
    )
    missing = [w for w in POSITIVE + NEGATIVE if w not in model]
    if missing:
        raise ValueError(f"Нет слов в модели: {', '.join(missing)}")

    result = nearest_nouns(model, POSITIVE, NEGATIVE, TOP_N)
    print("Комбинация: развалина + будущее")
    print("Десять ближайших существительных:")
    for rank, (word, score) in enumerate(result, start=1):
        print(f"{rank:2}. {word.removesuffix('_NOUN'):20} {score:.4f}")

    ranks = {word: rank for rank, (word, _) in enumerate(result, start=1)}
    for word in sorted(TARGETS):
        print(f"{word.removesuffix('_NOUN')}: место {ranks.get(word, 'вне десятки')}")
    if not TARGETS.issubset(ranks):
        raise AssertionError("Заданная пара не вошла в первую десятку")


if __name__ == "__main__":
    main()
