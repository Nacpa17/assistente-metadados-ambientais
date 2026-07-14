"""Métricas para avaliação de QA.

BLEU (sacrebleu), ROUGE-L (rouge-score), F1 token-level, exact match e
acurácia de recusa para perguntas negativas.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable


_REFUSAL_MARKERS = [
    "não está disponível",
    "não consta",
    "não contêm",
    "não contém",
    "não há registro",
    "não disponível",
    "não tenho essa informação",
    "não foi informado",
]


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s.,:/;\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(s: str) -> list[str]:
    return normalize(s).split()


def token_f1(pred: str, gold: str) -> float:
    pt, gt = tokenize(pred), tokenize(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pt)
    recall = n_common / len(gt)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def is_refusal(text: str) -> bool:
    n = normalize(text)
    return any(normalize(m) in n for m in _REFUSAL_MARKERS)


def corpus_bleu(predictions: Iterable[str], references: Iterable[str]) -> float:
    try:
        import sacrebleu
    except ImportError:
        return float("nan")
    preds = list(predictions)
    refs = [list(references)]
    if not preds:
        return 0.0
    return float(sacrebleu.corpus_bleu(preds, refs).score)


def avg_rouge_l(predictions: list[str], references: list[str]) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0


def aggregate(predictions: list[str], references: list[str],
              kinds: list[str] | None = None) -> dict[str, float]:
    n = len(predictions)
    if n == 0:
        return {}
    em = sum(exact_match(p, r) for p, r in zip(predictions, references)) / n
    f1 = sum(token_f1(p, r) for p, r in zip(predictions, references)) / n
    out = {
        "n": n,
        "exact_match": round(em, 4),
        "token_f1": round(f1, 4),
        "bleu": round(corpus_bleu(predictions, references), 2),
        "rouge_l": round(avg_rouge_l(predictions, references), 4),
    }
    if kinds:
        neg_idx = [i for i, k in enumerate(kinds) if k == "negative"]
        if neg_idx:
            correct = sum(is_refusal(predictions[i]) for i in neg_idx)
            out["refusal_accuracy"] = round(correct / len(neg_idx), 4)
            out["n_negative"] = len(neg_idx)
        by_kind: dict[str, list[float]] = {}
        for i, k in enumerate(kinds):
            by_kind.setdefault(k, []).append(
                token_f1(predictions[i], references[i]))
        out["token_f1_by_kind"] = {
            k: round(sum(vs) / len(vs), 4) for k, vs in by_kind.items()}
    return out
