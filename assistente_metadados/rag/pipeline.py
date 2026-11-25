"""Orquestra a recuperação + montagem do prompt — usado por avaliação, API e CLI.

Fica separado do código com torch para poder ser testado e reutilizado em
qualquer lugar: quem tem o modelo só precisa passar a função que conta tokens.

    rag = RagPipeline.from_config(cfg)          # lê a seção `rag:` do YAML
    res = rag.build_prompt(system, pergunta, history,
                           count_tokens=lambda s: len(tok(s).input_ids))
    res.prompt   -> texto para o modelo
    res.sources  -> fichas usadas (para mostrar "Fontes" ao usuário)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Any, Callable, Iterable

from assistente_metadados.common.prompts import DEFAULT_RAG_INSTRUCTIONS, build_rag_prompt
from assistente_metadados.rag.context import build_context, fit_to_budget
from assistente_metadados.rag.index import Hit, Index


@dataclass
class RagSettings:
    index_path: str = "data/rag/index.json"
    top_k: int = 3
    max_context_chars: int = 5000
    max_input_tokens: int = 3072      # LLaMA 2 tem 4096; sobra p/ a resposta
    min_score: float = 0.0
    expand_pt_en: bool = True
    instructions: str = DEFAULT_RAG_INSTRUCTIONS

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None, **overrides: Any) -> "RagSettings":
        raw = dict((cfg or {}).get("rag") or {})
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in raw.items() if k in known})
        s.top_k = int(s.top_k)
        s.max_context_chars = int(s.max_context_chars)
        s.max_input_tokens = int(s.max_input_tokens)
        s.min_score = float(s.min_score)
        return s


@dataclass
class RagResult:
    prompt: str
    hits: list[Hit]                    # fichas que entraram no prompt
    retrieved: list[Hit]               # tudo que a busca devolveu
    retrieval_s: float
    context: str = ""
    query_used: str = ""

    @property
    def sources(self) -> list[dict[str, Any]]:
        return sources_payload(self.hits)


def sources_payload(hits: Iterable[Hit]) -> list[dict[str, Any]]:
    """Resumo serializável das fichas (para JSON da API e da avaliação)."""
    out = []
    for n, h in enumerate(hits, 1):
        d = h.doc
        out.append({
            "n": n,
            "id": d.id,
            "kind": d.kind,
            "title": d.title,
            "doi": d.meta.get("doi"),
            "score": round(h.score, 3),
            "exact_match": bool(h.boost),
        })
    return out


def _cites_dataset(hits: list[Hit]) -> bool:
    return any(h.boost and h.doc.kind == "dataset" for h in hits)


def _last_user_message(history: list[dict] | None) -> str | None:
    for msg in reversed(history or []):
        if msg.get("role") == "user" and (msg.get("content") or "").strip():
            return msg["content"].strip()
    return None


class RagPipeline:
    def __init__(self, index: Index, settings: RagSettings | None = None):
        self.index = index
        self.settings = settings or RagSettings()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None, **overrides: Any) -> "RagPipeline":
        s = RagSettings.from_config(cfg, **overrides)
        return cls(Index.load(s.index_path), s)

    # ------------------------------------------------------------------ #
    def retrieve(self, question: str,
                 history: list[dict] | None = None) -> tuple[list[Hit], str]:
        """Busca pela pergunta. Em perguntas de continuação ("E qual o DOI
        dele?"), que não citam o dataset, busca de novo junto com a pergunta
        anterior do usuário — e usa esse resultado se ele identificar um
        dataset (ou, quando a pergunta atual não casa com nada, qualquer
        identificador)."""
        s = self.settings
        kw = dict(k=s.top_k, min_score=s.min_score, expand=s.expand_pt_en)
        hits = self.index.search(question, **kw)
        prev = _last_user_message(history)
        if prev and not _cites_dataset(hits):
            combined = f"{prev} {question}"
            hits2 = self.index.search(combined, **kw)
            if _cites_dataset(hits2) or (
                    not any(h.boost for h in hits)
                    and any(h.boost for h in hits2)):
                return hits2, combined
        return hits, question

    def build_prompt(self, system: str, question: str,
                     history: list[dict] | None = None,
                     count_tokens: Callable[[str], int] | None = None) -> RagResult:
        s = self.settings
        t0 = time.perf_counter()
        hits, query = self.retrieve(question, history)
        dt = time.perf_counter() - t0
        hist = list(history or [])
        ctx_holder: dict[str, str] = {}

        def make(hs: list[Hit]) -> tuple[str, list[Hit]]:
            ctx, used = build_context(hs, question, max_chars=s.max_context_chars)
            ctx_holder["ctx"] = ctx
            return build_rag_prompt(system, hist, ctx, question,
                                    instructions=s.instructions), used

        if count_tokens is not None:
            prompt, used = fit_to_budget(make, hits, count_tokens,
                                         s.max_input_tokens)
        else:
            prompt, used = make(hits)
        return RagResult(prompt=prompt, hits=used, retrieved=hits,
                         retrieval_s=dt, context=ctx_holder.get("ctx", ""),
                         query_used=query)
