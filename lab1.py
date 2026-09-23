import argparse
from pathlib import Path
from typing import Iterator

from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3


TARGET_POS = {"NOUN", "ADJF"}
MIN_PARSE_SCORE = 0.01


def has_same_grammar(first: object, second: object) -> bool:
    first_tag = first.tag
    second_tag = second.tag
    if first_tag.number != second_tag.number or first_tag.case != second_tag.case:
        return False
    return first_tag.number == "plur" or first_tag.gender == second_tag.gender


def matching_pair(first_options: list[object], second_options: list[object]) -> tuple[object, object] | None:
    candidates = []

    for first in first_options:
        for second in second_options:
            if (
                first.score >= MIN_PARSE_SCORE
                and second.score >= MIN_PARSE_SCORE
                and first.tag.POS in TARGET_POS
                and second.tag.POS in TARGET_POS
                and "Apro" not in first.tag
                and "Apro" not in second.tag
                and has_same_grammar(first, second)
            ):
                candidates.append((first, second))

    if not candidates:
        return None

    return max(candidates, key=lambda pair: pair[0].score * pair[1].score)

def find_matching_pairs(text: str) -> Iterator[tuple[str, str]]:
    analyzer = pymorphy3.MorphAnalyzer()

    for sentence in sent_tokenize(text, language="russian"):
        analyses = [analyzer.parse(token) for token in word_tokenize(sentence)]
        for first_options, second_options in zip(analyses, analyses[1:]):
            pair = matching_pair(first_options, second_options)
            if pair:
                first, second = pair
                yield first.normal_form, second.normal_form


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    text = args.input_file.read_text(encoding="utf-8")

    for first_lemma, second_lemma in find_matching_pairs(text):
        print(first_lemma, second_lemma)


if __name__ == "__main__":
    main()
