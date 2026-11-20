#!/usr/bin/env python
# coding: utf-8

# # Лабораторная 5: собака управляет логитами
# 
# Модель Qwen3-1.7B. Собака запрещает ненавистные ей слова и поддерживает слова про собачью жизнь.
# 

# In[1]:

# Для запуска через python lab5_export.py нужно убрать строку get_ipython() и установить зависимости отдельно.
# get_ipython().run_line_magic('pip', 'install -q "transformers==5.21.0" "pymorphy3==2.0.6"')


# In[2]:


import torch
#assert torch.cuda.is_available(), "Выберите GPU в настройках среды Colab."
#print(torch.cuda.get_device_name(0))


# ## 1. Настройки
# 
# banned_lemmas — начальные формы; склонения и спряжения добавит pymorphy3.
# 
# banned_words — точные слова.
# 
# favored_words — точные слова для усиления.
# 

# In[ ]:


rules = {'banned_words': ['мяу', 'мур','кыс','царапать','царапаться','шипеть'],
 'banned_lemmas': ['кот', 'кошка', 'котёнок', 'кошачий', 'мурлыкать', 'мурлыканье', 'когтеточка','котик','киса','сосед'],
 'favored_words': ['прогулка', 'косточка', 'гав'],
 'boost': 2.5,
 'completion_bonus': 6.0,
 'warmup_tokens': 20}

SEEDS = [5, 35, 18]
MAX_NEW_TOKENS = 100


# ## 2. Подготовка словаря и токенов
# 
# Служебные функции получают словоформы и восстанавливают байты токенов Qwen, чтобы запрет работал и при разбиении одной буквы на несколько токенов. 

# In[ ]:


import json
import math
import re
from functools import lru_cache
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, set_seed

MODEL_NAME = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PROMPT = "Напиши короткий рассказ, который начинается со звонка в дверь."
ROOT = Path.cwd()
SEPARATORS = ".,!?;:()[]{}\"'«»…"


def is_separator(char):
    return char.isspace() or char in SEPARATORS


def word_pattern(words):
    return re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, words)) + r")(?![\w-])", re.I)


def validate_rules(rules):
    for key in ("banned_words", "banned_lemmas", "favored_words"):
        words = rules.get(key, [])
        if not isinstance(words, list) or any(
            not isinstance(word, str) or re.fullmatch(r"[^\W\d_]+", word) is None
            for word in words
        ):
            raise ValueError(f"{key}: нужен список отдельных слов из букв")
    if set(map(str.casefold, rules["banned_words"])) & set(map(str.casefold, rules["favored_words"])):
        raise ValueError("Слово нельзя одновременно запрещать и поддерживать")
    for key in ("boost", "completion_bonus"):
        if not isinstance(rules[key], (int, float)) or not math.isfinite(rules[key]) or rules[key] < 0:
            raise ValueError(f"{key}: нужно конечное неотрицательное число")
    warmup = rules.get("warmup_tokens", 20)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
        raise ValueError("warmup_tokens: нужно положительное целое число")


@lru_cache(maxsize=1)
def morphology():
    from pymorphy3 import MorphAnalyzer
    return MorphAnalyzer()


@lru_cache(maxsize=128)
def lemma_forms(word):
    parsed = morphology().parse(word)[0]
    forms = {form.word.casefold() for form in parsed.lexeme}
    return frozenset(forms | {form.replace("ё", "е") for form in forms})


def expand_banned_words(rules):
    validate_rules(rules)
    banned = set(map(str.casefold, rules["banned_words"]))
    for lemma in rules.get("banned_lemmas", []):
        banned.update(lemma_forms(lemma.casefold()))
    validate_rules({**rules, "banned_words": sorted(banned)})
    return sorted(banned)


def qwen_token_bytes(tokenizer):
    decoder_type = json.loads(tokenizer.backend_tokenizer.to_str())["decoder"]["type"]
    if decoder_type != "ByteLevel":
        raise ValueError("Этот обработчик рассчитан на ByteLevel-токенизатор Qwen3")
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    decoder = {chr(byte): byte for byte in visible}
    for index, byte in enumerate(byte for byte in range(256) if byte not in visible):
        decoder[chr(256 + index)] = byte
    special = set(tokenizer.all_special_ids)
    return [b"" if i in special else bytes(decoder[char] for char in token)
            for i, token in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer)))))]




# ## 3. Обработчик и функции запуска
# 
# В DogLogitsProcessor.__call__ меняются оценки следующего токена:
# 
# 1. Завершение запрещённого слова получает -inf.
# 2. Начало или продолжение собачьего слова получает бонус.
# 3. После готового слова сохраняется разделитель; повторно это слово не усиливается.
# 
# 

# In[ ]:


class DogLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, prompt_length, rules):
        self.tokenizer, self.prompt_length = tokenizer, prompt_length
        self.rules = {"warmup_tokens": 20, **rules}
        self.pieces = qwen_token_bytes(tokenizer)
        self.banned = banned = expand_banned_words(rules)
        self.banned_pattern = word_pattern(banned) if banned else None
        endings = tuple({word.casefold()[offset:] for word in banned for offset in range(len(word))})
        self.ban_candidates = []
        for token, piece in enumerate(self.pieces):
            decoded = piece.decode("utf-8", errors="ignore").casefold()
            if piece and (decoded.startswith(endings)
                          or any(word.casefold() in decoded for word in banned)
                          or (banned and 0x80 <= piece[0] < 0xC0)):
                self.ban_candidates.append(token)
        self.tail_size = max((len(word.encode("utf-8")) for word in banned), default=0) + 4
        self.favored = [(word_pattern([word]), [tokenizer.encode(form, add_special_tokens=False)
                         for case in (word.lower(), word.capitalize(), word.upper())
                         for form in (case, " " + case)]) for word in rules["favored_words"]]
        self.boundary_ids = [i for i, piece in enumerate(self.pieces)
                             if piece and is_separator(piece.decode("utf-8", errors="replace")[0])]
        self.boundary_ids.append(tokenizer.eos_token_id)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        scores[:, len(self.pieces):] = -float("inf")
        for row, tokens in enumerate(input_ids[:, self.prompt_length:].tolist()):
            raw = b"".join(self.pieces[token] for token in tokens)
            text = raw.decode("utf-8", errors="replace")

            if any(any(match.end() == len(text) for match in pattern.finditer(text))
                   for pattern, _ in self.favored):
                allowed = scores[row, self.boundary_ids].clone()
                scores[row].fill_(-float("inf"))
                scores[row, self.boundary_ids] = allowed
            seen = any(pattern.search(text) for pattern, _ in self.favored)
            pressure = 0.0 if seen else min(1.0, len(tokens) / self.rules["warmup_tokens"])
            candidates = {}
            for pattern, variants in self.favored:
                if pattern.search(text):
                    continue
                for ids in variants:
                    for size in range(min(len(tokens), len(ids) - 1), -1, -1):
                        if size and tokens[-size:] != ids[:size]:
                            continue
                        before = self.decode(tokens[:-size] if size else tokens)
                        prefix = self.decode(ids[:max(1, size)])
                        if not prefix.startswith(" ") and before and not is_separator(before[-1]):
                            continue
                        token = ids[size]
                        candidates[token] = max(candidates.get(token, False), size > 0)
                        break

            completing = any(candidates.values())
            ids = [token for token, partial in candidates.items()
                   if (partial or not completing) and torch.isfinite(scores[row, token])]
            if ids:
                gap = (scores[row].max() - scores[row, ids]).clamp_min(0)
                bonus = (gap + self.rules["completion_bonus"] if completing
                         else self.rules["boost"] + pressure * gap)
                scores[row, ids] += bonus

            tail = raw[-self.tail_size:]
            previous_length = len(tail.decode("utf-8", errors="ignore"))
            for token in self.ban_candidates:
                candidate = (tail + self.pieces[token]).decode("utf-8", errors="ignore")
                if any(match.end() > previous_length for match in self.banned_pattern.finditer(candidate)):
                    scores[row, token] = -float("inf")
        return scores


def generate(model, tokenizer, inputs, seed, max_new_tokens, processor=None, probe=None):
    set_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.9, top_k=0, top_p=1.0, repetition_penalty=1.0,
            logits_processor=LogitsProcessorList([] if processor is None else [processor]),
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_logits=probe is not None,
        )
    length = inputs["input_ids"].shape[1]
    ids = output.sequences[0, length:].tolist()
    checks = []
    if probe is not None:
        for step, scores in enumerate(output.logits):
            checks = check_logits(output.sequences[:, :length + step], scores.float(), probe)
            if checks:
                for row in checks:
                    row.update(step=step + 1, context=probe.decode(ids[:step]))
                break
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False), ids, checks


def check_logits(input_ids, before, processor, limit=5):
    after = processor(input_ids, before)
    ids = torch.tensor(processor.ban_candidates, device=before.device, dtype=torch.long)
    ids = ids[torch.isfinite(before[0, ids]) & torch.isneginf(after[0, ids])]
    ids = ids[before[0, ids].argsort(descending=True)[:limit]]
    p_before, p_after = (scores.div(0.9).softmax(dim=-1) for scores in (before, after))
    return [dict(token=processor.decode([token]), token_id=token,
                 logit_before=before[0, token].item(), logit_after="-inf",
                 probability_before=p_before[0, token].item(),
                 probability_after=p_after[0, token].item()) for token in ids.tolist()]


def run(rules, seeds=(5, 7, 18), max_new_tokens=100, cache_dir=None, output=None):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens должен быть положительным")
    banned = expand_banned_words(rules)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    print(f"Модель: {MODEL_NAME}; {device}, {dtype}", flush=True)
    loading = dict(revision=MODEL_REVISION, cache_dir=cache_dir or ROOT / ".hf_cache")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **loading)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, **loading).to(device).eval()
    if inputs["input_ids"].shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Промпт и продолжение превышают контекст модели")
    processor = DogLogitsProcessor(tokenizer, inputs["input_ids"].shape[1], rules)
    report = dict(model=MODEL_NAME, revision=MODEL_REVISION, prompt=PROMPT,
                  formatted_prompt=formatted, enable_thinking=False, rules=rules,
                  banned_forms=banned, seeds=list(seeds), max_new_tokens=max_new_tokens,
                  temperature=0.9, top_k=0, top_p=1.0, repetition_penalty=1.0,
                  device=device, dtype=str(dtype), torch_version=torch.__version__,
                  transformers_version=transformers.__version__, results=[], ban_check=[])
    output = Path(output or ROOT / "results.json")
    for seed in seeds:
        for name, active in (("baseline", None), ("dog", processor)):
            text, ids, checks = generate(model, tokenizer, inputs, seed, max_new_tokens, active,
                                         probe=processor if active is None else None)
            report["ban_check"].extend(dict(seed=seed, **row) for row in checks)
            found = {key: [word for word in words if word_pattern([word]).search(text)]
                     for key, words in (("banned_words", banned), ("favored_words", rules["favored_words"]))}
            report["results"].append(dict(seed=seed, variant=name, text=text, token_ids=ids, **found))
            print(f"\nSeed {seed}; {'без обработчика' if active is None else 'с собакой'}:\n{text}", flush=True)
            print(f"Найдены слова: {found}", flush=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {output}")
    return report



# ## 4. Генерация ответов
# 
# baseline — без обработчика, dog — с ним. Для каждой пары совпадают модель, промпт, seed и sampling.
# 
# 

# In[12]:


report = run(rules, seeds=SEEDS, max_new_tokens=MAX_NEW_TOKENS)


# ## 5. Что встретилось в ответах

# In[15]:


import pandas as pd
summary = pd.DataFrame([
    {"seed": row["seed"], "вариант": row["variant"],
     "запрещённые слова в ответе": ", ".join(row["banned_words"]),
     "желательные слова в ответе": ", ".join(row["favored_words"])}
    for row in report["results"]
])
print(summary.to_string(index=False))


# ## 6. Оценки запрета
# 
# 
# 

# In[16]:


table = pd.DataFrame(report["ban_check"]).rename(columns={
    "token": "токен", "token_id": "ID", "logit_before": "оценка Qwen до", "step": "шаг",
    "context": "контекст обычного ответа",
    "logit_after": "оценка после запрета", "probability_before": "вероятность до",
    "probability_after": "вероятность после",
})
if table.empty:
    print("В этих ответах не встретился шаг с доступным запрещённым завершением. По ним запрет оценить не удалось.")
else:
    table["токен"] = table["токен"].map(repr)
    table["контекст обычного ответа"] = table["контекст обычного ответа"].str[-60:]
    print(table.to_string(index=False, formatters={
        "вероятность до": "{:.3e}".format,
        "вероятность после": "{:.1f}".format,
    }))

