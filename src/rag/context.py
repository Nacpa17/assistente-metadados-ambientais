"""Monta o bloco de CONTEXTO que vai no prompt a partir dos documentos achados.

- Cada documento vira uma ficha numerada "[1] ...", com rótulos em português.
- Nas fichas de dataset, as variáveis citadas na pergunta sobem para o topo da
  lista e ganham a descrição (útil para "O que significa a variável X em Y?").
- Há dois orçamentos: em CARACTERES (aqui, sem depender de tokenizer) e em
  TOKENS (fit_to_budget, que recebe a função de contagem do tokenizer real e
  remove as fichas menos relevantes até o prompt caber).
"""

from __future__ import annotations

from typing import Callable, Sequence

from src.prompts.generator import cap_join
from src.rag.documents import Doc
from src.rag.index import Hit
from src.rag.text import phrase, tokenize

KIND_LABEL = {
    "dataset": "Ficha de dataset",
    "glossary": "Glossário",
    "variable": "Variável (índice do corpus)",
    "facility": "Observatório (índice do corpus)",
    "corpus": "Estatísticas do corpus",
    "dqr": "Relatório de qualidade da ARM (DQR)",
    "site": "Sítio ARM (índice da Data Discovery)",
}

MAX_VAR_NAMES = 20
MAX_VAR_DESCS = 5


def _variable_lines(doc: Doc, query: str) -> list[str]:
    variables = doc.meta.get("variables") or []
    if not variables:
        return []
    qp = phrase(query)
    qt = set(tokenize(query))

    def relevance(v: dict) -> float:
        name = v["name"]
        p = phrase(name)
        if p.strip() and p in qp:
            return 10.0                       # nome exato citado na pergunta
        return float(len(qt & set(tokenize(name))))

    scored = [(relevance(v), j, v) for j, v in enumerate(variables)]
    relevant = [v for s, _, v in sorted(scored, key=lambda x: (-x[0], x[1]))
                if s > 0]
    names = [v["name"] for v in relevant]
    names += [v["name"] for v in variables if v["name"] not in set(names)]
    out = [f"Variáveis medidas ({len(variables)}): "
           f"{cap_join(names, MAX_VAR_NAMES)}"]
    for v in relevant[:MAX_VAR_DESCS]:
        unit = f" (unidade: {v['units']})" if v.get("units") else ""
        if v.get("description") or unit:
            out.append(f"Variável {v['name']}: "
                       f"{v.get('description') or '(sem descrição)'}{unit}")
    return out


def render_doc(doc: Doc, query: str = "") -> list[str]:
    """Linhas da ficha (sem numeração), adaptadas à pergunta."""
    lines = list(doc.lines)
    if doc.kind == "dataset":
        n_core = int(doc.meta.get("n_core", len(lines)))
        lines = lines[:n_core] + _variable_lines(doc, query) + lines[n_core:]
    return lines


def _cut(lines: list[str], max_chars: int, min_lines: int = 2) -> list[str]:
    """Corta a ficha em fronteira de linha (mantém ao menos `min_lines`)."""
    out: list[str] = []
    total = 0
    for ln in lines:
        if out and len(out) >= min_lines and total + len(ln) + 1 > max_chars:
            break
        out.append(ln)
        total += len(ln) + 1
    if total > max_chars and out:              # linha única gigante
        out[-1] = out[-1][: max(0, max_chars - (total - len(out[-1])))] + "…"
    return out


def build_context(hits: Sequence[Hit], query: str = "",
                  max_chars: int = 5000, first_doc_chars: int = 3000,
                  other_doc_chars: int = 1500) -> tuple[str, list[Hit]]:
    """Retorna (texto do contexto, fichas efetivamente incluídas)."""
    blocks: list[str] = []
    used: list[Hit] = []
    total = 0
    for n, h in enumerate(hits, 1):
        limit = first_doc_chars if n == 1 else other_doc_chars
        body = _cut(render_doc(h.doc, query), limit)
        block = f"[{len(used) + 1}] {KIND_LABEL.get(h.doc.kind, h.doc.kind)}\n" \
                + "\n".join(body)
        if used and total + len(block) + 2 > max_chars:
            break
        if not used and len(block) > max_chars:
            block = block[:max_chars].rstrip() + "…"
        blocks.append(block)
        used.append(h)
        total += len(block) + 2
    return "\n\n".join(blocks), used


def fit_to_budget(make_prompt: Callable[[list[Hit]], tuple[str, list[Hit]]],
                  hits: Sequence[Hit], count_tokens: Callable[[str], int],
                  max_tokens: int) -> tuple[str, list[Hit]]:
    """Remove as fichas do fim até o prompt caber em `max_tokens`.

    Se nem sem contexto couber (histórico enorme), devolve a versão sem
    fichas — o corte à esquerda do tokenizer cuida do resto.
    """
    current = list(hits)
    while True:
        prompt, used = make_prompt(current)
        if count_tokens(prompt) <= max_tokens or not current:
            return prompt, used
        current = list(used[:-1]) if len(used) < len(current) else current[:-1]
