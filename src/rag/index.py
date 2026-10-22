"""Índice de busca: BM25 em Python puro + reforço por identificador exato.

Ranqueamento em dois níveis:
1. REFORÇO POR IDENTIFICADOR: se a pergunta contém, inteiro, um identificador
   de um documento (nome completo do dataset, código "1twrmr", sufixo de DOI,
   termo do glossário, nome de variável, sigla do observatório), o documento
   ganha um reforço proporcional ao tamanho do identificador (nomes longos são
   mais específicos que siglas). É o que garante a recuperação EXATA — o ponto
   fraco do fine-tuning puro na fase 2.
2. BM25 (Robertson; k1=1.5, b=0.75) desempata e cobre perguntas sem
   identificador ("onde encontro dados de umidade em SGP?"). Termos em
   português são expandidos para o inglês dos metadados (umidade -> humidity)
   com peso reduzido, via data/rag/expansoes_pt_en.yaml.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.common.io import load_json, load_yaml, save_json
from src.rag.documents import Doc
from src.rag.text import (STOPWORDS, is_acronym, normalize, phrase,
                          strip_accents, tokenize)

INDEX_VERSION = 1
EXPANSION_WEIGHT = 0.5
DATASET_TIEBREAK = 0.5
INTENT_TIEBREAK = 0.25
AGGREGATE_KINDS = {"variable", "facility", "site"}
# Perguntas que PROCURAM DADOS ("onde encontro / quais datasets / preciso de
# dados de..."), em oposição a pedir uma definição ("o que é X?").
WHERE_INTENT = re.compile(
    r"\b(onde|quais datasets|em quais|quantos|quantas|medem|mede|medid\w*|"
    r"encontr\w*|disponive\w*|recomend\w*|existem|preciso|procuro|busco|"
    r"quero|dados (de|sobre|para)|coordenad\w*|facilit\w*)\b")
# Nessas perguntas, definições do glossário valem metade no BM25: o
# glossário está em português e casaria melhor com a pergunta do que as
# fichas de dataset (em inglês), escondendo justamente os dados procurados.
GLOSSARY_FACTOR_DATA_INTENT = 0.5
# Perguntas sobre QUALIDADE: trazem os DQRs do dataset citado (se houver
# catálogo extra com DQRs, veja src/rag/extra.py).
QUALITY_INTENT = re.compile(
    r"\b(qualidade|dqrs?|problemas?|suspeit\w*|incorret\w*|confiave\w*|"
    r"confiavel|falhas?|defeitos?|anomalias?|ausentes?|faltantes?|"
    r"quality|issues?)\b")


@dataclass
class Hit:
    doc: Doc
    score: float          # BM25
    boost: float          # reforço por identificador exato
    matched_keys: list[str]

    @property
    def rank_key(self) -> tuple[float, float]:
        return (self.boost, self.score)


def load_expansions(path: str | Path | None) -> dict[str, list[str]]:
    """Lê o dicionário PT->EN (chaves e valores normalizados, sem acento)."""
    if not path or not Path(path).exists():
        return {}
    data = load_yaml(path) or {}
    out: dict[str, list[str]] = {}
    for k, vs in (data.get("termos") or {}).items():
        key = normalize(str(k)).strip()
        vals = [normalize(str(v)).strip() for v in (vs or []) if str(v).strip()]
        if key and vals:
            out[key] = vals
    return out


def _usable_key(k: str) -> tuple[str, bool] | None:
    """(forma canônica, sensível_a_maiúsculas) de um identificador, ou None.

    Descarta chaves de 1 caractere e palavras comuns ("e", "a", "T"), que
    casariam com qualquer pergunta em português. Exceção: sigla em MAIÚSCULAS
    que coincide com palavra comum ("AOS" x "aos") vira chave que só casa
    se a pergunta também a escrever em maiúsculas.
    """
    p = phrase(k)
    toks = p.split()
    if not toks:
        return None
    if len(toks) == 1:
        if len(toks[0]) <= 1:
            return None
        if toks[0] in STOPWORDS:
            if is_acronym(strip_accents(k).strip()):
                return phrase(k, lower=False), True
            return None
    return p, False


class Index:
    def __init__(self, docs: list[Doc], expansions: dict[str, list[str]] | None = None,
                 k1: float = 1.5, b: float = 0.75,
                 info: dict[str, Any] | None = None):
        self.docs = docs
        self.expansions = expansions or {}
        self.k1 = k1
        self.b = b
        self.info = info or {}
        self.by_id = {d.id: d for d in docs}
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for i, d in enumerate(self.docs):
            toks = tokenize(" ".join([d.title, d.search_text, " ".join(d.keys)]))
            self.doc_len.append(len(toks))
            for t, c in Counter(toks).items():
                self.postings[t].append((i, c))
        self.dqr_by_class: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(self.docs):
            if d.kind == "dqr":
                for c in d.meta.get("classes") or []:
                    self.dqr_by_class[str(c).lower()].append(i)
        n = max(1, len(self.docs))
        self.avgdl = (sum(self.doc_len) / n) or 1.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

        # identificadores exatos: (forma canônica, doc, peso, sensível a
        # maiúsculas), agrupados pelo primeiro token para checar só os
        # candidatos de cada pergunta.
        self.keys: list[tuple[str, int, float, bool]] = []
        self.keys_by_first: dict[str, list[tuple[str, int, float, bool]]] = \
            defaultdict(list)
        for i, d in enumerate(self.docs):
            seen: set[str] = set()
            for k in d.keys:
                uk = _usable_key(k)
                if not uk or uk[0] in seen:
                    continue
                p, cs = uk
                seen.add(p)
                entry = (p, i, 1.0 + len(p.split()), cs)
                self.keys.append(entry)
                self.keys_by_first[p.split()[0].lower()].append(entry)

    # ------------------------------------------------------------------ #
    def query_terms(self, query: str, expand: bool = True) -> dict[str, float]:
        terms: dict[str, float] = Counter(tokenize(query))
        if expand and self.expansions:
            for t in list(terms):
                for e in self.expansions.get(t, []):
                    for et in tokenize(e):
                        terms[et] = max(terms.get(et, 0.0), EXPANSION_WEIGHT)
        return dict(terms)

    def bm25(self, terms: dict[str, float]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for t, w in terms.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                denom = tf + k1 * (1 - b + b * self.doc_len[i] / avgdl)
                scores[i] += w * idf * tf * (k1 + 1) / denom
        return scores

    def key_matches(self, query: str) -> dict[int, tuple[float, list[str]]]:
        qp = phrase(query)
        qp_cs = phrase(query, lower=False)
        found: dict[int, tuple[float, list[str]]] = {}
        if not qp:
            return found
        for t in dict.fromkeys(qp.split()):     # ordem estável (set muda por processo)
            for p, i, w, cs in self.keys_by_first.get(t, ()):
                if p in (qp_cs if cs else qp):
                    boost, ks = found.get(i, (0.0, []))
                    found[i] = (boost + w, ks + [p.strip()])
        # Desempates entre documentos citados com o mesmo peso:
        # - o DATASET citado vem primeiro (mais específico; maior orçamento);
        # - "onde/quais datasets/quantos..." prefere os índices agregados
        #   (variável, observatório); "o que é/defina..." prefere o glossário.
        prefer = (AGGREGATE_KINDS if WHERE_INTENT.search(normalize(query))
                  else {"glossary"})  # mesma regra de intenção do search()
        for i in found:
            kind = self.docs[i].kind
            extra = (DATASET_TIEBREAK if kind == "dataset"
                     else INTENT_TIEBREAK if kind in prefer else 0.0)
            if extra:
                boost, ks = found[i]
                found[i] = (boost + extra, ks)
        return found

    def search(self, query: str, k: int = 5, kinds: Iterable[str] | None = None,
               min_score: float = 0.0, expand: bool = True,
               focus: bool = True) -> list[Hit]:
        """Top-k documentos para a pergunta.

        focus=True: se a pergunta cita um DATASET pelo identificador exato,
        devolve só os documentos citados explicitamente (o dataset, termos do
        glossário, variáveis...). Isso tira do contexto os "quase iguais" —
        ex.: 1twrmr x 30twrmr têm o mesmo título e DOIs diferentes, e um
        distrator desses no prompt induz o modelo a copiar o DOI errado.
        """
        kinds_set = set(kinds) if kinds else None
        scores = self.bm25(self.query_terms(query, expand=expand))
        keys = self.key_matches(query)
        data_intent = bool(WHERE_INTENT.search(normalize(query)))
        hits: list[Hit] = []
        for i in set(scores) | set(keys):
            d = self.docs[i]
            if kinds_set and d.kind not in kinds_set:
                continue
            boost, matched = keys.get(i, (0.0, []))
            s = scores.get(i, 0.0)
            if data_intent and d.kind == "glossary" and not boost:
                s *= GLOSSARY_FACTOR_DATA_INTENT
            if boost == 0.0 and s < min_score:
                continue
            hits.append(Hit(doc=d, score=s, boost=boost, matched_keys=matched))
        hits.sort(key=lambda h: h.rank_key, reverse=True)
        if focus and any(h.boost and h.doc.kind == "dataset" for h in hits):
            hits = [h for h in hits if h.boost]
        if self.dqr_by_class and QUALITY_INTENT.search(normalize(query)):
            hits = self._with_class_dqrs(hits)
        return hits[:k]

    def _with_class_dqrs(self, hits: list[Hit]) -> list[Hit]:
        """Pergunta de qualidade sobre um dataset citado: os DQRs do produto
        entram logo depois da ficha do dataset (os mais recentes primeiro).
        Entram por ASSOCIAÇÃO, não por identificador (boost 0)."""
        cited = [h for h in hits if h.boost and h.doc.kind == "dataset"]
        existing = {h.doc.id: h for h in hits}
        dqr_ids: set[str] = set()
        extra: list[Hit] = []
        for h in cited:
            cls = str(h.doc.meta.get("short_code") or "").lower().split(".")[0]
            for i in self.dqr_by_class.get(cls, []):
                d = self.docs[i]
                if d.id in dqr_ids:
                    continue
                dqr_ids.add(d.id)
                # um DQR que a busca já achou mantém seu Hit (score/boost),
                # mas entra no bloco ordenado por data como os demais
                extra.append(existing.get(d.id) or Hit(
                    doc=d, score=0.0, boost=0.0, matched_keys=[]))
        if not extra:
            return hits
        extra.sort(key=lambda h: str(h.doc.meta.get("start") or ""),
                   reverse=True)
        cited_ids = {h.doc.id for h in cited}
        rest = [h for h in hits
                if h.doc.id not in cited_ids and h.doc.id not in dqr_ids]
        return cited + extra + rest

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        kinds = Counter(d.kind for d in self.docs)
        save_json({
            "version": INDEX_VERSION,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {"k1": self.k1, "b": self.b},
            "info": {**self.info, "n_docs": len(self.docs),
                     "n_docs_by_kind": dict(kinds)},
            "expansions": self.expansions,
            "docs": [d.to_dict() for d in self.docs],
        }, path, indent=None)

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Índice RAG não encontrado em {p}. Gere com:\n"
                f"  python -m src.rag.build_index --json-dir data/raw_json "
                f"--out {p}")
        data = load_json(p)
        if data.get("version") != INDEX_VERSION:
            raise ValueError(f"Versão de índice incompatível em {p} "
                             f"({data.get('version')} != {INDEX_VERSION}); "
                             f"reconstrua com src.rag.build_index.")
        prm = data.get("params", {})
        return cls([Doc.from_dict(d) for d in data["docs"]],
                   expansions=data.get("expansions") or {},
                   k1=float(prm.get("k1", 1.5)), b=float(prm.get("b", 0.75)),
                   info=data.get("info") or {})
