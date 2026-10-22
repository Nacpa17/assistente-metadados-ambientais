"""Métricas para avaliação de QA.

BLEU (sacrebleu), ROUGE-L (rouge-score), F1 token-level, exact match e
acurácia de recusa para perguntas negativas.

Fase 3 (RAG):
- marcadores de recusa corrigidos: 2 das 6 frases de recusa usadas no treino
  (src/prompts/add_negatives.py) e a resposta-ouro "fora do escopo" do
  benchmark NÃO eram reconhecidas, o que subestimava refusal_accuracy;
- false_refusal_rate: fração das perguntas COM resposta em que o modelo
  recusou (risco típico do RAG quando a busca falha);
- contains_gold: a resposta-ouro aparece dentro da resposta do modelo
  (normalizada). Para fatos curtos (DOI, data, formato), "O DOI é X." está
  correto mas zera o exact match; esta é a "accuracy" usada em estudos de
  recordação de fatos como Kandpal et al. (2023).
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
    # fase 3: frases do próprio treino/benchmark que não eram reconhecidas
    "não está descrito",   # "Esse detalhe não está descrito nos metadados..."
    "não encontro",        # "Não encontro essa informação nos metadados..."
    "fora do escopo",      # gold: "Essa pergunta está fora do escopo..."
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


def contains_gold(pred: str, gold: str) -> float:
    """1.0 se a resposta-ouro (normalizada) está contida na predição."""
    g = normalize(gold)
    return float(bool(g) and g in normalize(pred))


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
    # perguntas COM resposta (todas, se não houver categorias)
    pos_idx = [i for i in range(n)
               if not kinds or kinds[i] != "negative"]
    if pos_idx:
        out["contains_gold"] = round(sum(
            contains_gold(predictions[i], references[i])
            for i in pos_idx) / len(pos_idx), 4)
        out["false_refusal_rate"] = round(sum(
            is_refusal(predictions[i]) for i in pos_idx) / len(pos_idx), 4)
        out["n_positive"] = len(pos_idx)
    if kinds:
        neg_idx = [i for i, k in enumerate(kinds) if k == "negative"]
        if neg_idx:
            correct = sum(is_refusal(predictions[i]) for i in neg_idx)
            out["refusal_accuracy"] = round(correct / len(neg_idx), 4)
            out["n_negative"] = len(neg_idx)
        by_kind: dict[str, list[int]] = {}
        for i, k in enumerate(kinds):
            by_kind.setdefault(k, []).append(i)
        out["token_f1_by_kind"] = {
            k: round(sum(token_f1(predictions[i], references[i])
                         for i in ix) / len(ix), 4)
            for k, ix in by_kind.items()}
        detail: dict[str, dict[str, float]] = {}
        for k, ix in by_kind.items():
            d = {"n": len(ix),
                 "token_f1": out["token_f1_by_kind"][k],
                 "exact_match": round(sum(exact_match(predictions[i],
                                                      references[i])
                                          for i in ix) / len(ix), 4)}
            if k == "negative":
                d["refusal_accuracy"] = round(
                    sum(is_refusal(predictions[i]) for i in ix) / len(ix), 4)
            else:
                d["contains_gold"] = round(
                    sum(contains_gold(predictions[i], references[i])
                        for i in ix) / len(ix), 4)
            detail[k] = d
        out["by_kind"] = detail
    return out
