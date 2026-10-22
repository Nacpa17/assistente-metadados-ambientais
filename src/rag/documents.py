"""Transforma os metadados em DOCUMENTOS recuperáveis (as "fichas" do RAG).

Cinco tipos de documento, cada um respondendo a um tipo de pergunta do
benchmark/dataset de treino:

- dataset   : uma ficha por JSON-LD (DOI, período, local, instrumento, ...)
              -> perguntas descritivas ("Qual o DOI de X?")
- glossary  : uma entrada do glossário (curado + auto)
              -> "O que é reflectivity?", "O que significa KAZR?"
- variable  : variável -> em quais datasets aparece (índice invertido)
              -> "Quais datasets medem rh?", "Quantos datasets medem temp?"
- facility  : observatório -> datasets, temas, coleta ativa
              -> "Quais datasets existem para Southern Great Plains?"
- corpus    : estatísticas globais
              -> "Qual a variável mais medida no corpus?"

Os rótulos e frases das fichas reaproveitam as MESMAS formulações das
respostas geradas por src/prompts/generator.py (ex.: o período sai como
"1998-04-01 até o presente (coleta contínua)"). Assim o modelo ajustado na
fase 2 encontra no contexto exatamente o formato que aprendeu a produzir.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from src.prompts.generator import Record, cap_join, fmt_examples, parse_year

from src.rag.text import doi_suffix

DESC_MAX_CHARS = 450


@dataclass
class Doc:
    id: str
    kind: str                      # dataset | glossary | variable | facility | corpus
    title: str
    lines: list[str]               # linhas fixas da ficha (rótulo: valor)
    search_text: str               # texto indexado pelo BM25
    keys: list[str] = field(default_factory=list)   # identificadores exatos
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Doc":
        return cls(**{k: d[k] for k in
                      ("id", "kind", "title", "lines", "search_text",
                       "keys", "meta")})


def _shorten(s: str | None, max_chars: int = DESC_MAX_CHARS) -> str | None:
    if not s:
        return None
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    dot = cut.rfind(". ")
    return cut[: dot + 1] if dot > max_chars * 0.5 else cut.rstrip() + "…"


def _facility_answer(r: Record) -> str | None:
    """Mesma regra do gerador: prefere o detalhe do site quando diferente."""
    if not r.facility:
        return None
    if r.site_detail and r.site_detail != r.facility:
        return r.site_detail
    return r.facility


def _creators_text(r: Record, cap: int = 8) -> str | None:
    if not r.creators:
        return None
    parts = [
        f"{c['name']}"
        + (f" ({c['role']}, {c['affiliation']})" if c["role"]
           else f" ({c['affiliation']})")
        for c in r.creators[:cap]
    ]
    if len(r.creators) > cap:
        parts.append(f"entre outros ({len(r.creators)} no total)")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Fichas por dataset
# --------------------------------------------------------------------------- #
def dataset_doc(r: Record, doc_id: str) -> Doc:
    # Ordem pensada para cortes: fatos curtos e mais perguntados primeiro
    # (bloco "core"); a ficha insere as variáveis relevantes logo após ele.
    L: list[str] = [f"Dataset: {r.name}"]
    if r.short_code and r.short_code != r.name:
        L.append(f"Código (datastream): {r.short_code}")
    if r.doi:
        L.append(f"DOI / acesso: {r.doi}")
    if r.period_display:
        L.append(f"Período de coleta: {r.period_display}")
        if r.ongoing:
            L.append(f"Status da coleta: ativa (iniciou em {r.period_start} "
                     f"e continua)")
        elif r.period_end:
            L.append(f"Status da coleta: encerrada (de {r.period_start} a "
                     f"{r.period_end})")
    if r.lat is not None:
        loc = f"Latitude: {r.lat}, Longitude: {r.lon}"
        if r.site_detail:
            loc += f" ({r.site_detail})"
        L.append(f"Localização: {loc}")
    if (fac := _facility_answer(r)):
        L.append(f"Observatório (facility): {fac}")
        if r.facility_url:
            L.append(f"Página do observatório: {r.facility_url}")
    if r.instrument:
        L.append(f"Instrumento: {r.instrument}")
        if r.instrument_url:
            L.append(f"Detalhes do instrumento: {r.instrument_url}")
    if r.data_format:
        L.append(f"Formato dos dados: {r.data_format}")
    if r.dist_url:
        L.append(f"Acesso direto aos dados: "
                 f"{r.dist_name or 'ARM Data Discovery'}: {r.dist_url}")
    if r.free is True:
        L.append("Acesso gratuito: sim (requer login ARM para download)")
    n_core = len(L)
    if (d := _shorten(r.description)):
        L.append(f"Descrição: {d}")
    if r.citation:
        L.append(f"Citação recomendada: {r.citation}")
    for label, val in (("Provedor (provider)", r.provider),
                       ("Publicador (publisher)", r.publisher),
                       ("Organização de origem (sourceOrganization)",
                        r.source_org)):
        if val:
            L.append(f"{label}: {val}")
    if (cr := _creators_text(r)):
        L.append(f"Criadores: {cr}")
    mentors = [c["name"] for c in r.creators
               if c["role"] and "mentor" in c["role"].lower()]
    if mentors:
        L.append(f"Instrument mentors: {', '.join(mentors)}")
    if r.themes:
        L.append(f"Palavras-chave / temas: {cap_join(r.themes, 15)}")
    if r.date_published:
        L.append(f"Data de publicação: {r.date_published}")
    if r.date_modified:
        L.append(f"Última atualização dos metadados: {r.date_modified}")
    if r.catalog_name:
        L.append("Catálogo: " + r.catalog_name
                 + (f" ({r.catalog_url})" if r.catalog_url else ""))
    if r.enc_format:
        L.append(f"Formato de codificação dos metadados: {r.enc_format}")
    if r.enc_url:
        L.append(f"Codificação dos metadados: {r.enc_url}")

    var_names = [v["name"] for v in r.variables]
    search = " ".join(filter(None, [
        r.name, r.short_code, r.description, r.facility, r.site_detail,
        r.instrument, r.data_format, r.provider, r.publisher, r.source_org,
        r.doi, " ".join(r.keywords), " ".join(var_names),
        " ".join(v["description"] for v in r.variables if v["description"]),
        " ".join(c["name"] for c in r.creators if c.get("name")),
    ]))

    keys = [r.name]
    if r.short_code and r.short_code != r.name:
        keys.append(r.short_code)
    if (ds := doi_suffix(r.doi)):
        keys.append(ds)

    return Doc(
        id=doc_id, kind="dataset", title=r.name, lines=L,
        search_text=search, keys=keys,
        meta={"dataset": r.name, "short_code": r.short_code, "doi": r.doi,
              "facility": r.facility, "ongoing": r.ongoing, "n_core": n_core,
              # lista completa: a ficha escolhe as relevantes por pergunta
              "variables": r.variables},
    )


# --------------------------------------------------------------------------- #
# Glossário
# --------------------------------------------------------------------------- #
def glossary_doc(e: dict[str, Any], doc_id: str) -> Doc | None:
    term = str(e.get("term") or "").strip()
    definition = (e.get("definition") or "").strip()
    if not term or not definition:
        return None
    aliases = [str(a).strip() for a in (e.get("aliases") or []) if str(a).strip()]
    expansao = (e.get("expansao") or "").strip()
    unit = (e.get("unit") or "").strip()
    uso = (e.get("uso") or "").strip()
    source = e.get("source") if e.get("source") in ("auto", "dod") else "curated"
    standard_name = (e.get("standard_name") or "").strip()

    head = f"Termo: {term}"
    if expansao:
        head += f" (sigla de {expansao})"
    L = [head]
    if source == "curated":
        L.append(f"Definição: {definition}")
    elif source == "dod":
        L.append(f'Definição (nome longo oficial no DOD da ARM, em inglês): '
                 f'"{definition}"')
    else:
        L.append(f'Definição (descrição em inglês extraída dos metadados ARM): '
                 f'"{definition}"')
    if uso:
        L.append(f"Para que serve: {uso}")
    if unit:
        if e.get("unit_source") == "dod":
            alts = [str(a) for a in e.get("unit_alternatives") or []]
            L.append(f"Unidade (DOD oficial da ARM): {unit}"
                     + (f"; alguns produtos usam {', '.join(alts)}" if alts
                        else ""))
        else:
            L.append(f"Unidade típica: {unit}")
    if standard_name:
        L.append(f"Nome padrão CF (standard_name): {standard_name}")
    if aliases:
        L.append(f"Outras grafias nos datastreams: {', '.join(aliases[:10])}")
    if e.get("category"):
        L.append(f"Categoria: {e['category']}")

    search = " ".join([term, expansao, " ".join(aliases), definition, uso,
                       standard_name, str(e.get("category") or "")])
    return Doc(id=doc_id, kind="glossary", title=term, lines=L,
               search_text=search, keys=[term] + aliases,
               meta={"term": term, "expansao": expansao or None,
                     "source": source})


# --------------------------------------------------------------------------- #
# Documentos agregados (reaproveitam os índices invertidos do gerador)
# --------------------------------------------------------------------------- #
def variable_docs(records: list[Record], n_examples: int = 5) -> list[Doc]:
    by_key: dict[str, list[Record]] = defaultdict(list)
    casing: dict[str, Counter] = defaultdict(Counter)
    descs: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        seen: set[str] = set()
        for v in r.variables:
            k = v["name"].lower()
            casing[k][v["name"]] += 1
            if v["description"]:
                descs[k][v["description"]] += 1
            if k not in seen:
                by_key[k].append(r)
                seen.add(k)

    docs = []
    for k, recs in sorted(by_key.items()):
        name = casing[k].most_common(1)[0][0]
        L = [f"Variável: {name}",
             f"Medida em {len(recs)} dataset(s) do corpus analisado."]
        if descs.get(k):
            L.append(f"Descrição nos metadados: {descs[k].most_common(1)[0][0]}")
        L.append(f"Exemplos de datasets: {fmt_examples(recs, n_examples)}.")
        facs = sorted({r.facility for r in recs if r.facility})
        if facs:
            L.append(f"Observatórios: {cap_join(facs, 6)}")
        search = " ".join([name, " ".join(descs.get(k, {}).keys())])
        docs.append(Doc(
            id=f"var:{k}", kind="variable", title=f"Variável {name}",
            lines=L, search_text=search, keys=list(casing[k].keys()),
            meta={"variable": name, "n_datasets": len(recs)}))
    return docs


def _facility_keys(fac: str) -> list[str]:
    """"Southern Great Plains (SGP), Central Facility, Lamont, OK (C1)" ->
    [nome completo, "Southern Great Plains", "SGP"]. Códigos com dígito
    (ex.: "C1") não viram chave: são ambíguos entre observatórios."""
    keys = [fac]
    base = fac.split("(")[0].strip().rstrip(",").strip()
    if base and base != fac:
        keys.append(base)
    keys.extend(re.findall(r"\(([A-Za-z]{2,6})\)", fac))
    return keys


def facility_docs(records: list[Record], n_examples: int = 5) -> list[Doc]:
    by_fac: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.facility:
            by_fac[r.facility].append(r)

    docs = []
    for fac, recs in sorted(by_fac.items()):
        active = [r for r in recs if r.ongoing]
        # dict.fromkeys: sem repetidos e em ordem estável (set muda por processo)
        themes = Counter(t for r in recs for t in dict.fromkeys(r.themes))
        instrs = sorted({r.instrument for r in recs if r.instrument})
        L = [f"Observatório: {fac}",
             f"Datasets no corpus: {len(recs)}. Exemplos: "
             f"{fmt_examples(recs, n_examples)}."]
        if active:
            L.append(f"Em coleta ativa: {len(active)} dataset(s). Exemplos: "
                     f"{fmt_examples(active, n_examples)}.")
        else:
            L.append("Em coleta ativa: nenhum dataset consta como coleta "
                     "ativa nos metadados analisados.")
        if themes:
            L.append("Temas mais frequentes: " + ", ".join(
                f"{t} ({c})" for t, c in themes.most_common(10)))
        if instrs:
            L.append(f"Instrumentos: {cap_join(instrs, 10, '; ')}")
        keys = _facility_keys(fac)
        search =" ".join([fac, " ".join(themes), " ".join(instrs),
                           " ".join(dict.fromkeys(r.site_detail or ""
                                              for r in recs))])
        docs.append(Doc(
            id=f"fac:{fac.lower()}", kind="facility", title=f"Observatório {fac}",
            lines=L, search_text=search, keys=keys,
            meta={"facility": fac, "n_datasets": len(recs),
                  "n_active": len(active)}))
    return docs


def corpus_doc(records: list[Record]) -> Doc | None:
    if not records:
        return None
    var_counter: Counter = Counter()
    for r in records:
        # dict.fromkeys (não set): empates em most_common ficam na ordem de
        # aparição, a mesma do gerador — a ficha e a resposta de treino
        # citam a mesma variável "mais frequente"
        var_counter.update(list(dict.fromkeys(v["name"] for v in r.variables)))
    years = [y for y in (parse_year(r.period_start) for r in records
                         if r.period_start) if y]
    n_active = sum(1 for r in records if r.ongoing)
    facs = Counter(r.facility for r in records if r.facility)

    L = [f"Corpus analisado: {len(records)} datasets indexados "
         f"(metadados JSON-LD do repositório ARM)."]
    if var_counter:
        top = var_counter.most_common(5)
        v0, c0 = top[0]
        L.append(f"Variável mais frequentemente medida no corpus: {v0}, "
                 f"presente em {c0} dataset(s).")
        L.append("Cinco variáveis mais comuns no corpus: "
                 + ", ".join(f"{v} ({c})" for v, c in top))
    if years:
        L.append(f"Cobertura temporal aproximada: os datasets analisados "
                 f"iniciam entre {min(years)} e {max(years)}; {n_active} "
                 f"permanecem em coleta contínua.")
    if facs:
        L.append("Observatórios com mais datasets: " + ", ".join(
            f"{f} ({c})" for f, c in facs.most_common(8)))
    # Só vocabulário de pergunta AGREGADA: palavras como "período" ou
    # "coleta" ficam de fora para esta ficha não disputar espaço nas
    # perguntas sobre um dataset específico.
    search = ("corpus total analisado analisados indexados frequente "
              "frequentemente comum comuns cobertura estatisticas ranking")
    return Doc(id="corpus", kind="corpus", title="Visão geral do corpus",
               lines=L, search_text=search, keys=[],
               meta={"n_datasets": len(records)})


def build_documents(records: list[Record],
                    glossary_entries: list[dict[str, Any]]) -> list[Doc]:
    docs: list[Doc] = []
    used: Counter = Counter()
    for r in records:
        base = f"ds:{(r.short_code or r.name).lower()}"
        used[base] += 1
        doc_id = base if used[base] == 1 else f"{base}#{used[base]}"
        docs.append(dataset_doc(r, doc_id))

    gused: Counter = Counter()
    for e in glossary_entries:
        base = f"gl:{str(e.get('term') or '').lower()}"
        gused[base] += 1
        doc_id = base if gused[base] == 1 else f"{base}#{gused[base]}"
        if (d := glossary_doc(e, doc_id)):
            docs.append(d)

    docs.extend(variable_docs(records))
    docs.extend(facility_docs(records))
    if (c := corpus_doc(records)):
        docs.append(c)
    return docs
