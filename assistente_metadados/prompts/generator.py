"""Gerador do dataset de pares pergunta-resposta — versão 3.1.

v3 (mantido):
- Extração rica dos JSON-LD (description, facility, instrument, formato,
  status de coleta, mentores, datas...).
- Pools de templates named/contextual (sem frases quebradas).
- Coleta contínua ("1998-04-01/..") e datas placeholder tratadas.
- Citações sem HTML; spans site/facility viram localização textual.
- QA de corpus com índices invertidos: variável→datasets, facility→datasets,
  (facility, tema)→recomendação com DOI, coleta ativa por facility.
- Estratégia keep_all no mix (corrige dataset minúsculo do v2).

v3.1 (novo):
- GLOSSÁRIO: lê data/glossary/glossary_full.json (curado + auto) e gera
  perguntas de definição ("O que é mean_doppler_velocity?"), uso ("Para que
  serve?"), unidade, expansão de siglas ("O que significa KAZR?"), aliases
  ("MeanDopplerVelocity e mean_doppler_velocity são a mesma coisa?") e
  definição+localização ("O que é X e onde encontro essa medição?").

Uso:
    python -m assistente_metadados.glossary.build --json-dir data/raw_json \
        --out data/glossary/glossary_full.json      # (uma vez)
    python -m assistente_metadados.prompts.generator --config configs/dataset_v3.yaml
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from assistente_metadados.common.io import load_json, load_yaml, setup_logger, write_jsonl

log = setup_logger("prompts.generator")

GENERIC_KEYWORDS = {
    "arm", "ornl", "adc", "arm research facility", "climate",
    "atmospheric data",
}

MAX_SANE_YEAR = 2100


# --------------------------------------------------------------------------- #
# Limpeza de texto
# --------------------------------------------------------------------------- #
def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_description(s: str, max_chars: int = 700) -> str | None:
    if not isinstance(s, str):
        return None
    s = re.sub(r"\[caption[^\]]*\].*?\[/caption\]", " ", s, flags=re.DOTALL)
    s = s.replace("View More", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    if len(s) > max_chars:
        cut = s[:max_chars]
        dot = cut.rfind(". ")
        s = (cut[: dot + 1] if dot > max_chars * 0.5 else cut.rstrip())
    return s or None


def parse_year(s: str) -> int | None:
    m = re.match(r"^\s*(\d{4})", s or "")
    if not m:
        return None
    y = int(m.group(1))
    return y if 1900 <= y <= MAX_SANE_YEAR else None


# --------------------------------------------------------------------------- #
# Registro estruturado por dataset
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    name: str
    short_code: str
    doi: str | None = None
    description: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    ongoing: bool = False
    lat: float | None = None
    lon: float | None = None
    facility: str | None = None
    facility_url: str | None = None
    site_detail: str | None = None
    instrument: str | None = None
    instrument_url: str | None = None
    variables: list[dict] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    creators: list[dict] = field(default_factory=list)
    provider: str | None = None
    publisher: str | None = None
    source_org: str | None = None
    data_format: str | None = None
    free: bool | None = None
    date_created: str | None = None
    date_modified: str | None = None
    date_published: str | None = None
    catalog_name: str | None = None
    catalog_url: str | None = None
    citation: str | None = None
    dist_name: str | None = None
    dist_url: str | None = None
    dist_desc: str | None = None
    enc_format: str | None = None
    enc_url: str | None = None

    @property
    def period_display(self) -> str | None:
        if not self.period_start:
            return None
        if self.ongoing:
            return f"{self.period_start} até o presente (coleta contínua)"
        if self.period_end:
            return f"{self.period_start} a {self.period_end}"
        return self.period_start

    def period_days(self) -> int | None:
        def _p(s: str) -> datetime | None:
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y"):
                try:
                    return datetime.strptime(s.strip(), fmt)
                except ValueError:
                    continue
            return None
        if not self.period_start:
            return None
        a = _p(self.period_start)
        b = datetime.now() if self.ongoing else (
            _p(self.period_end) if self.period_end else None
        )
        if not a or not b:
            return None
        return max(0, (b - a).days)


def _org_name(d: dict, key: str) -> str | None:
    v = d.get(key)
    if isinstance(v, dict):
        n = v.get("name")
        if isinstance(n, str) and n.strip():
            return n.strip()
    return None


def parse_record(d: dict[str, Any]) -> Record | None:
    name = (d.get("name") or "").strip()
    if not name:
        return None
    short = name.split(" - ")[-1].strip() if " - " in name else name

    rec = Record(name=name, short_code=short)

    url = d.get("url")
    rec.doi = url.strip() if isinstance(url, str) and url.strip() else None
    rec.description = clean_description(d.get("description"))

    t = d.get("temporalCoverage")
    if isinstance(t, str) and "/" in t:
        a, b = (x.strip() for x in t.split("/", 1))
        rec.period_start = a or None
        if b in ("..", ""):
            rec.ongoing = True
        else:
            y = parse_year(b)
            if y is None:
                rec.ongoing = True
            else:
                rec.period_end = b

    geo = d.get("spatialCoverage")
    if isinstance(geo, list) and geo and isinstance(geo[0], dict):
        g = geo[0].get("geo") or {}
        try:
            rec.lat = float(g.get("latitude"))
            rec.lon = float(g.get("longitude"))
        except (TypeError, ValueError):
            rec.lat = rec.lon = None

    for entry in d.get("about") or []:
        if not isinstance(entry, dict):
            continue
        n = (entry.get("name") or "").strip()
        u = (entry.get("url") or "").strip() or None
        if n.startswith("Facility Information:"):
            rec.facility = n.split(":", 1)[1].strip()
            rec.facility_url = u
        elif n.startswith("Instrument Information:"):
            rec.instrument = n.split(":", 1)[1].strip()
            rec.instrument_url = u

    cit = d.get("citation")
    if isinstance(cit, str) and cit.strip():
        site = re.search(r"id='site'>([^<]*)<", cit)
        fac = re.search(r"id='facility'>([^<]*)<", cit)
        parts = [m.group(1).strip() for m in (site, fac)
                 if m and m.group(1).strip()]
        rec.site_detail = ", ".join(parts) or None
        rec.citation = strip_html(cit)

    for v in d.get("variableMeasured") or []:
        if isinstance(v, dict) and v.get("name"):
            rec.variables.append({
                "name": v["name"].strip(),
                "description": (v.get("description") or "").strip(),
            })

    var_names = {v["name"].lower() for v in rec.variables}
    short_l = short.lower()
    for k in d.get("keywords") or []:
        if not isinstance(k, str) or not k.strip():
            continue
        k = k.strip()
        rec.keywords.append(k)
        kl = k.lower()
        if (kl not in GENERIC_KEYWORDS and kl not in var_names
                and "_" not in k and len(k) > 3
                and kl not in short_l and short_l not in kl):
            rec.themes.append(k)

    for c in d.get("creator") or []:
        if not isinstance(c, dict):
            continue
        aff = c.get("affiliation")
        rec.creators.append({
            "name": c.get("name") or "N/A",
            "affiliation": (aff.get("name") if isinstance(aff, dict) else None)
                           or "Desconhecida",
            "role": (c.get("description") or "").strip() or None,
        })

    rec.provider = _org_name(d, "provider")
    rec.publisher = _org_name(d, "publisher")
    rec.source_org = _org_name(d, "sourceOrganization")

    dist = d.get("distribution") or {}
    if isinstance(dist, dict):
        rec.dist_name = dist.get("name")
        rec.dist_url = dist.get("url")
        rec.dist_desc = dist.get("description")
        rec.data_format = dist.get("encodingFormat")

    if isinstance(d.get("isAccessibleForFree"), bool):
        rec.free = d["isAccessibleForFree"]

    rec.date_created = d.get("dateCreated")
    rec.date_modified = d.get("dateModified")
    rec.date_published = d.get("datePublished")

    cat = d.get("includedInDataCatalog") or {}
    if isinstance(cat, dict):
        rec.catalog_name = cat.get("name")
        rec.catalog_url = cat.get("url")

    enc = d.get("encoding") or {}
    if isinstance(enc, dict):
        rec.enc_format = enc.get("encodingFormat")
        rec.enc_url = enc.get("contentUrl")

    return rec


# --------------------------------------------------------------------------- #
# Utilidades de listagem
# --------------------------------------------------------------------------- #
def cap_join(items: list[str], n: int, sep: str = ", ") -> str:
    if len(items) <= n:
        return sep.join(items)
    return sep.join(items[:n]) + f", entre outros ({len(items)} no total)"


def fmt_examples(recs: list["Record"], n: int) -> str:
    parts = []
    for r in recs[:n]:
        parts.append(f"{r.name} (acesso: {r.doi})" if r.doi else r.name)
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Builder de QA por dataset
# --------------------------------------------------------------------------- #
class QABuilder:
    def __init__(self, cfg: dict[str, Any], rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.variants: dict[str, dict[str, list[str]]] = cfg["prompt_variants"]
        self.neg_named: list[str] = cfg["negative_templates"]["named"]
        self.neg_ctx: list[str] = cfg["negative_templates"]["contextual"]
        self.ambiguity: float = float(cfg.get("ambiguity_ratio", 0.3))
        self.list_cap: int = int(cfg.get("list_cap", 20))
        self.seen: set[tuple[str, str]] = set()

    def _pick(self, key: str, **fmt: Any) -> tuple[str, bool]:
        pools = self.variants.get(key)
        if not pools:
            raise KeyError(f"sem variantes para '{key}'")
        named = pools.get("named") or []
        ctx = pools.get("contextual") or []
        use_ctx = ctx and (not named or self.rng.random() < self.ambiguity)
        pool = ctx if use_ctx else named
        return self.rng.choice(pool).format(**fmt), bool(use_ctx)

    def _pair(self, kind: str, prompt: str, completion: str,
              dataset: str) -> dict | None:
        prompt = prompt.strip()
        completion = completion.strip()
        key = (prompt, completion)
        if key in self.seen or not prompt or not completion:
            return None
        self.seen.add(key)
        return {"prompt": prompt, "completion": completion,
                "meta": {"kind": kind, "dataset": dataset}}

    def _add(self, out: list, kind: str, key: str, completion: str,
             r: Record, **fmt: Any) -> None:
        prompt, _ = self._pick(key, name=r.name, **fmt)
        pair = self._pair(kind, prompt, completion, r.name)
        if pair:
            out.append(pair)

    # -------------------- descritivo -------------------- #
    def descriptive(self, r: Record) -> list[dict]:
        out: list[dict] = []
        A = self._add

        A(out, "descriptive_name", "name", r.name, r)
        if r.short_code and r.short_code != r.name:
            A(out, "descriptive_code", "instrument_code", r.short_code, r)

        if r.description:
            A(out, "descriptive_purpose", "purpose", r.description, r)

        if r.period_display:
            A(out, "descriptive_period", "period", r.period_display, r)
            status = (
                f"Sim. A coleta iniciou em {r.period_start} e continua ativa."
                if r.ongoing else
                f"Não. A coleta ocorreu de {r.period_start} a {r.period_end}."
            )
            A(out, "descriptive_status", "ongoing_status", status, r)

        if r.lat is not None:
            loc = f"Latitude: {r.lat}, Longitude: {r.lon}"
            if r.site_detail:
                loc += f" ({r.site_detail})"
            A(out, "descriptive_location", "location", loc, r)

        if r.facility:
            ans = r.facility
            if r.site_detail and r.site_detail != r.facility:
                ans = r.site_detail
            if r.facility_url:
                ans += f". Mais informações: {r.facility_url}"
            A(out, "descriptive_facility", "facility", ans, r)

        if r.instrument:
            ans = r.instrument
            if r.instrument_url:
                ans += f". Detalhes do instrumento: {r.instrument_url}"
            A(out, "descriptive_instrument", "instrument", ans, r)

        for field_key, val in (("provider", r.provider),
                               ("publisher", r.publisher),
                               ("sourceOrganization", r.source_org)):
            if val:
                A(out, f"descriptive_{field_key}", "organization", val, r,
                  field=field_key)

        if r.doi:
            A(out, "descriptive_doi", "doi", r.doi, r)
        if r.dist_url:
            ans = f"{r.dist_name or 'ARM Data Discovery'}: {r.dist_url}"
            if r.dist_desc:
                ans += f". {r.dist_desc}"
            A(out, "descriptive_direct_access", "direct_access", ans, r)
        if r.data_format:
            A(out, "descriptive_data_format", "data_format", r.data_format, r)
        if r.free is True:
            A(out, "descriptive_free", "free_access",
              "Sim, o dataset é de acesso gratuito (requer login ARM para "
              "download).", r)

        if r.creators:
            det = "; ".join(
                f"{c['name']}"
                + (f" ({c['role']}, {c['affiliation']})" if c["role"]
                   else f" ({c['affiliation']})")
                for c in r.creators
            )
            A(out, "descriptive_creators", "creators", det, r)
            mentors = [c["name"] for c in r.creators
                       if c["role"] and "mentor" in c["role"].lower()]
            if mentors:
                A(out, "descriptive_mentors", "mentors",
                  ", ".join(mentors), r)

        if r.themes:
            A(out, "descriptive_keywords", "keywords",
              cap_join(r.themes, 15), r)

        if r.variables:
            names = [v["name"] for v in r.variables]
            A(out, "descriptive_variables", "variables",
              cap_join(names, self.list_cap), r)
            for v in r.variables:
                if v["description"]:
                    A(out, "descriptive_variable_meaning", "variable_meaning",
                      v["description"], r, field=v["name"])

        for key, val, kind in (
            ("date_published", r.date_published, "descriptive_published"),
            ("date_modified", r.date_modified, "descriptive_modified"),
        ):
            if val:
                A(out, kind, key, val, r)

        if r.catalog_name:
            ans = r.catalog_name + (f" ({r.catalog_url})" if r.catalog_url else "")
            A(out, "descriptive_catalog", "catalog", ans, r)

        if r.citation:
            A(out, "descriptive_citation", "citation", r.citation, r)

        if r.enc_format:
            A(out, "descriptive_encoding_format", "encoding_format",
              r.enc_format, r)
        if r.enc_url:
            A(out, "descriptive_encoding_url", "encoding_url", r.enc_url, r)

        return out

    # -------------------- negativos -------------------- #
    FIELD_CHECKS = {
        "period": lambda r: r.period_start is not None,
        "location": lambda r: r.lat is not None,
        "keywords": lambda r: bool(r.themes),
        "variables": lambda r: bool(r.variables),
        "creators": lambda r: bool(r.creators),
        "citation": lambda r: bool(r.citation),
        "purpose": lambda r: bool(r.description),
        "instrument": lambda r: bool(r.instrument),
        "facility": lambda r: bool(r.facility),
    }

    def negatives(self, r: Record, fields: list[str]) -> list[dict]:
        out: list[dict] = []
        for f in fields:
            check = self.FIELD_CHECKS.get(f)
            if check is None or check(r):
                continue
            prompt, is_ctx = self._pick(f, name=r.name, field=f)
            tpl = self.rng.choice(self.neg_ctx if is_ctx else self.neg_named)
            ans = tpl.format(name=r.name)
            pair = self._pair(f"negative_{f}", prompt, ans, r.name)
            if pair:
                out.append(pair)
        return out


# --------------------------------------------------------------------------- #
# QA de corpus: índices invertidos + recomendações
# --------------------------------------------------------------------------- #
class CorpusQA:
    def __init__(self, records: list[Record], cfg: dict[str, Any],
                 rng: random.Random, builder: QABuilder):
        self.recs = records
        self.cfg = cfg.get("corpus_qa", {})
        self.tpl = cfg["corpus_prompts"]
        self.rng = rng
        self.b = builder
        self.n_ex = int(self.cfg.get("examples_per_answer", 3))

        self.by_var: dict[str, list[Record]] = defaultdict(list)
        self.by_var_desc: dict[str, list[tuple[Record, str]]] = defaultdict(list)
        self.by_facility: dict[str, list[Record]] = defaultdict(list)
        self.by_fac_theme: dict[tuple[str, str], list[Record]] = defaultdict(list)
        self.by_theme: dict[str, list[Record]] = defaultdict(list)

        for r in records:
            for v in r.variables:
                self.by_var[v["name"]].append(r)
                if v["description"]:
                    self.by_var_desc[v["description"]].append((r, v["name"]))
            # dict.fromkeys (e não set): tira repetidos mantendo a ordem do
            # registro. A ordem de um set de strings muda a cada processo
            # (PYTHONHASHSEED) e, depois do shuffle com seed fixa, mudava o
            # dataset gerado de uma execução para outra.
            themes = list(dict.fromkeys(r.themes))
            if r.facility:
                self.by_facility[r.facility].append(r)
                for th in themes:
                    self.by_fac_theme[(r.facility, th)].append(r)
            for th in themes:
                self.by_theme[th].append(r)

    def _pairs(self, kind: str, key: str, items, answer_fn,
               cap: int) -> list[dict]:
        out = []
        items = list(items)
        self.rng.shuffle(items)
        for it in items[:cap]:
            prompt_tpl = self.rng.choice(self.tpl[key])
            prompt, completion = answer_fn(prompt_tpl, it)
            pair = self.b._pair(kind, prompt, completion, "_corpus")
            if pair:
                out.append(pair)
        return out

    def build(self) -> list[dict]:
        out: list[dict] = []
        C = self.cfg

        def var_answer(tpl, item):
            var, recs = item
            ans = (f"A variável {var} é medida em {len(recs)} dataset(s). "
                   f"Exemplos: {fmt_examples(recs, self.n_ex)}.")
            return tpl.format(var=var), ans
        out += self._pairs("corpus_variable_where", "variable_where",
                           self.by_var.items(), var_answer,
                           int(C.get("per_variable_max", 4000)))

        def desc_answer(tpl, item):
            desc, pairs = item
            ex = "; ".join(f"{r.name} (variável {v})"
                           for r, v in pairs[:self.n_ex])
            ans = f"Essa medição está disponível em: {ex}."
            if len(pairs) > self.n_ex:
                ans += f" Total de {len(pairs)} dataset(s)."
            return tpl.format(desc=desc), ans
        out += self._pairs("corpus_vardesc_where", "var_desc_where",
                           self.by_var_desc.items(), desc_answer,
                           int(C.get("per_var_description_max", 4000)))

        def fac_answer(tpl, item):
            fac, recs = item
            ans = (f"Há {len(recs)} dataset(s) coletado(s) em {fac}. "
                   f"Exemplos: {fmt_examples(recs, self.n_ex)}.")
            return tpl.format(facility=fac), ans
        out += self._pairs("corpus_facility_datasets", "facility_datasets",
                           self.by_facility.items(), fac_answer,
                           int(C.get("per_facility_max", 500)))

        def reco_answer(tpl, item):
            (fac, theme), recs = item
            ans = (f"Para dados de {theme} em {fac}, recomenda-se: "
                   f"{fmt_examples(recs, self.n_ex)}."
                   + (f" Total de {len(recs)} dataset(s) correspondente(s)."
                      if len(recs) > self.n_ex else ""))
            return tpl.format(theme=theme, facility=fac), ans
        out += self._pairs("corpus_reco", "facility_theme_reco",
                           self.by_fac_theme.items(), reco_answer,
                           int(C.get("per_facility_theme_max", 3000)))

        def instr_wrap(tpl, item):
            (fac, theme), recs = item
            instrs = sorted({r.instrument for r in recs if r.instrument})
            if not instrs:
                return "", ""
            ans = (f"Instrumentos associados a {theme} em {fac}: "
                   f"{cap_join(instrs, self.n_ex, '; ')}.")
            return tpl.format(theme=theme, facility=fac), ans
        out += self._pairs("corpus_instrument_reco",
                           "facility_theme_instrument",
                           [it for it in self.by_fac_theme.items()
                            if any(r.instrument for r in it[1])],
                           instr_wrap,
                           int(C.get("per_facility_theme_max", 3000)))

        def ongoing_answer(tpl, item):
            fac, recs = item
            act = [r for r in recs if r.ongoing]
            if act:
                ans = (f"Em {fac}, {len(act)} dataset(s) permanecem em coleta "
                       f"ativa. Exemplos: {fmt_examples(act, self.n_ex)}.")
            else:
                ans = (f"Nos metadados analisados, nenhum dataset de {fac} "
                       f"consta como coleta ativa no momento.")
            return tpl.format(facility=fac), ans
        out += self._pairs("corpus_facility_ongoing", "facility_ongoing",
                           self.by_facility.items(), ongoing_answer,
                           int(C.get("per_facility_max", 500)))

        def count_answer(tpl, item):
            var, recs = item
            return (tpl.format(var=var),
                    f"{len(recs)} dataset(s) medem a variável {var}.")
        out += self._pairs("analytical_variable_count", "variable_count",
                           self.by_var.items(), count_answer,
                           int(C.get("per_variable_count_max", 1500)))

        var_counter = Counter({v: len(rs) for v, rs in self.by_var.items()})
        if var_counter:
            top = var_counter.most_common(5)
            v0, c0 = top[0]
            p = self.b._pair(
                "analytical_top_variable",
                self.rng.choice(self.tpl["global_top_variable"]),
                f"{v0}, presente em {c0} dataset(s).", "_corpus")
            if p:
                out.append(p)
            p = self.b._pair(
                "analytical_top5_variables",
                self.rng.choice(self.tpl["global_top5_variables"]),
                ", ".join(f"{v} ({c})" for v, c in top), "_corpus")
            if p:
                out.append(p)

        years = [parse_year(r.period_start) for r in self.recs
                 if r.period_start]
        years = [y for y in years if y]
        if years:
            n_active = sum(1 for r in self.recs if r.ongoing)
            ans = (f"Os datasets analisados iniciam entre {min(years)} e "
                   f"{max(years)}; {n_active} permanecem em coleta contínua "
                   f"(total de {len(self.recs)} datasets indexados).")
            p = self.b._pair(
                "analytical_time_range",
                self.rng.choice(self.tpl["global_time_range"]), ans, "_corpus")
            if p:
                out.append(p)

        return out


# --------------------------------------------------------------------------- #
# QA de GLOSSÁRIO (novo em v3.1)
# --------------------------------------------------------------------------- #
class GlossaryQA:
    """Gera perguntas de definição/uso/unidade/sigla a partir do glossário
    (curado + auto), com ligação opcional ao corpus ("onde encontro").
    """

    def __init__(self, entries: list[dict], cfg: dict[str, Any],
                 rng: random.Random, builder: QABuilder,
                 by_var: dict[str, list[Record]]):
        self.entries = entries
        self.cfg = cfg.get("glossary_qa", {})
        self.tpl = cfg["glossary_prompts"]
        self.rng = rng
        self.b = builder
        self.n_ex = int(self.cfg.get("examples_per_answer", 3))
        # índice case-insensitive do corpus para o join "onde encontro"
        self.var_ci: dict[str, tuple[str, list[Record]]] = {}
        for name, recs in by_var.items():
            self.var_ci.setdefault(name.lower(), (name, []))
            self.var_ci[name.lower()][1].extend(recs)

    def _corpus_hits(self, entry: dict) -> list[Record]:
        names = [entry["term"]] + list(entry.get("aliases") or [])
        seen: dict[str, Record] = {}
        for n in names:
            hit = self.var_ci.get(str(n).lower())
            if hit:
                for r in hit[1]:
                    seen[r.name] = r
        return list(seen.values())

    def build(self) -> list[dict]:
        out: list[dict] = []
        max_alias_pairs = int(self.cfg.get("max_alias_pairs", 300))
        join_where = bool(self.cfg.get("join_where", True))
        alias_budget = max_alias_pairs

        for e in self.entries:
            term = e["term"]
            definition = (e.get("definition") or "").strip()
            uso = (e.get("uso") or "").strip()
            unit = (e.get("unit") or "").strip()
            expansao = (e.get("expansao") or "").strip()
            curated = e.get("source") == "curated"
            unit_dod = e.get("unit_source") == "dod"

            if not definition:
                continue

            # 1) definição
            if curated:
                ans = definition
                if unit:
                    ans += f" Unidade típica: {unit}."
            elif e.get("source") == "dod":
                # nome longo oficial declarado no DOD (glossary.build --extra)
                ans = (f'Nos metadados oficiais da ARM (DOD), "{term}" tem o '
                       f'nome longo "{definition}".')
            else:
                # auto: honestidade sobre a fonte (descrição dos metadados)
                ans = (f'Nos metadados ARM, "{term}" é descrito como: '
                       f'"{definition}".')
            p = self.b._pair(
                "glossary_definition",
                self.rng.choice(self.tpl["definition"]).format(term=term),
                ans, "_glossario")
            if p:
                out.append(p)

            # 2) uso ("para que serve")
            if uso:
                p = self.b._pair(
                    "glossary_purpose",
                    self.rng.choice(self.tpl["purpose"]).format(term=term),
                    uso, "_glossario")
                if p:
                    out.append(p)

            # 3) unidade
            if unit:
                if unit_dod:
                    alts = [str(a) for a in e.get("unit_alternatives") or []]
                    if unit.lower().startswith(("adimensional", "1 (")):
                        core = f"{term} é adimensional (sem unidade física)"
                    else:
                        core = f"{term} é expresso em {unit}"
                    unit_ans = (f"Segundo os DODs oficiais da ARM, {core}"
                                + (f"; alguns produtos usam {', '.join(alts)}"
                                   if alts else "") + ".")
                else:
                    unit_ans = f"{term} é tipicamente expresso em {unit}."
                p = self.b._pair(
                    "glossary_unit",
                    self.rng.choice(self.tpl["unit"]).format(term=term),
                    unit_ans, "_glossario")
                if p:
                    out.append(p)

            # 4) expansão de sigla
            if expansao:
                p = self.b._pair(
                    "glossary_acronym",
                    self.rng.choice(self.tpl["acronym"]).format(term=term),
                    f"{term} significa {expansao}. {definition}",
                    "_glossario")
                if p:
                    out.append(p)

            # 5) aliases ("são a mesma coisa?")
            aliases = [a for a in (e.get("aliases") or []) if str(a).strip()]
            if aliases and alias_budget > 0 and curated:
                b_alias = str(self.rng.choice(aliases))
                short_def = definition.split(". ")[0].rstrip(".") + "."
                tpl = self.rng.choice(self.tpl["alias"])
                # Resposta coerente com a forma da pergunta.
                if "diferença" in tpl.lower():
                    ans = (f"Não há diferença de significado: {term} e "
                           f"{b_alias} designam a mesma grandeza — "
                           f"{short_def} O que muda é apenas a grafia usada "
                           f"em cada datastream.")
                else:
                    ans = (f"Sim — referem-se à mesma grandeza: {short_def} "
                           f"A grafia varia conforme o datastream.")
                p = self.b._pair(
                    "glossary_alias",
                    tpl.format(a=term, b=b_alias),
                    ans, "_glossario")
                if p:
                    out.append(p)
                    alias_budget -= 1

            # 6) definição + onde encontrar no corpus
            if join_where:
                hits = self._corpus_hits(e)
                if hits:
                    base = (definition if curated
                            else f'Nos metadados ARM: "{definition}".')
                    ans = (f"{base} No corpus analisado, aparece em "
                           f"{len(hits)} dataset(s). Exemplos: "
                           f"{fmt_examples(hits, self.n_ex)}.")
                    p = self.b._pair(
                        "glossary_where",
                        self.rng.choice(self.tpl["where"]).format(term=term),
                        ans, "_glossario")
                    if p:
                        out.append(p)

        return out


# --------------------------------------------------------------------------- #
# Multi-hop
# --------------------------------------------------------------------------- #
def build_multihop(records: list[Record], cfg: dict, rng: random.Random,
                   builder: QABuilder) -> list[dict]:
    target = int(cfg.get("multi_hop_pairs", 2000))
    out: list[dict] = []
    usable = [r for r in records
              if r.period_start or r.variables or r.facility]
    if len(usable) < 2:
        return out

    attempts = 0
    while len(out) < target and attempts < target * 6:
        attempts += 1
        a, b = rng.sample(usable, 2)

        mode = rng.choice(["period", "vars", "facility"])
        if mode == "period" and a.period_start and b.period_start:
            da, db = a.period_days(), b.period_days()
            if da and db and da != db:
                longer = a if da > db else b
                extra = " (em coleta contínua)" if longer.ongoing else ""
                p = builder._pair(
                    "multi_hop_period",
                    f"Entre os datasets {a.name} e {b.name}, qual cobre o "
                    f"período mais longo?",
                    f"{longer.name} cobre o período mais longo{extra}.",
                    f"{a.name}|{b.name}")
                if p:
                    out.append(p)
        elif mode == "vars" and a.variables and b.variables:
            va = {v["name"] for v in a.variables}
            vb = {v["name"] for v in b.variables}
            common = sorted(va & vb)
            ans = (f"Variáveis em comum: {cap_join(common, 8)}." if common
                   else "Não há variáveis em comum entre os dois datasets.")
            p = builder._pair(
                "multi_hop_variables",
                f"Quais variáveis {a.name} e {b.name} compartilham?",
                ans, f"{a.name}|{b.name}")
            if p:
                out.append(p)
        elif mode == "facility" and a.facility and b.facility:
            same = a.facility == b.facility
            ans = (f"Sim, ambos foram coletados em {a.facility}." if same
                   else f"Não. {a.name} foi coletado em {a.facility}; "
                        f"{b.name}, em {b.facility}.")
            p = builder._pair(
                "multi_hop_facility",
                f"Os datasets {a.name} e {b.name} foram coletados no mesmo "
                f"observatório?",
                ans, f"{a.name}|{b.name}")
            if p:
                out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Mistura final — keep_all
# --------------------------------------------------------------------------- #
def assemble(buckets: dict[str, list[dict]], cfg: dict,
             rng: random.Random) -> list[dict]:
    mix = cfg.get("mix", {})
    strategy = mix.get("strategy", "keep_all")

    if strategy == "keep_all":
        neg_ratio = float(mix.get("negative_max_ratio", 0.15))
        non_neg = [p for k, v in buckets.items() if k != "negative" for p in v]
        neg = list(buckets.get("negative", []))
        max_neg = int(len(non_neg) * neg_ratio / max(1e-9, 1 - neg_ratio))
        if len(neg) > max_neg:
            rng.shuffle(neg)
            neg = neg[:max_neg]
        final = non_neg + neg
        rng.shuffle(final)
        return final

    props = {k: float(v) for k, v in mix.items()
             if k in buckets and isinstance(v, (int, float))}
    sizes = [len(buckets[k]) / p for k, p in props.items() if p > 0]
    if not sizes:
        return []
    total = int(min(sizes) * sum(props.values()))
    final: list[dict] = []
    for k, p in props.items():
        items = list(buckets[k])
        rng.shuffle(items)
        final.extend(items[: int(round(total * p / sum(props.values())))])
    rng.shuffle(final)
    return final


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(cfg_path: str) -> Path:
    cfg = load_yaml(cfg_path)
    rng = random.Random(cfg.get("seed", 42))
    builder = QABuilder(cfg, rng)

    input_dir = Path(cfg["input_json_dir"])
    files = sorted(input_dir.glob("*.json"))
    log.info("Lendo %d arquivos JSON de %s", len(files), input_dir)

    records: list[Record] = []
    for p in files:
        try:
            rec = parse_record(load_json(p))
            if rec:
                records.append(rec)
        except Exception as e:
            log.warning("Falha em %s: %s", p.name, e)
    log.info("Registros válidos: %d", len(records))

    desc: list[dict] = []
    neg: list[dict] = []
    for r in records:
        desc.extend(builder.descriptive(r))
        neg.extend(builder.negatives(r, cfg.get("fields_for_negatives", [])))

    corpus_qa = CorpusQA(records, cfg, rng, builder)
    corpus = corpus_qa.build()

    # ---- glossário (v3.1) ----
    glossary: list[dict] = []
    gpath = Path(cfg.get("glossary_path", ""))
    if gpath.exists():
        gdata = load_json(gpath)
        entries = gdata.get("entries", [])
        glossary = GlossaryQA(entries, cfg, rng, builder,
                              corpus_qa.by_var).build()
        log.info("Glossário: %d entradas (%d curadas, %d auto) => %d pares",
                 len(entries), gdata.get("n_curated", 0),
                 gdata.get("n_auto", 0), len(glossary))
    else:
        log.warning("Glossário %s não encontrado; pulando QA de glossário. "
                    "Rode: python -m assistente_metadados.glossary.build --json-dir %s "
                    "--out %s", gpath, input_dir, gpath)

    multi = build_multihop(records, cfg, rng, builder)

    log.info("Buckets — descritivo: %d | negativo: %d | corpus/analítico: %d "
             "| glossário: %d | multi-hop: %d",
             len(desc), len(neg), len(corpus), len(glossary), len(multi))

    final = assemble(
        {"descriptive": desc, "negative": neg, "corpus": corpus,
         "glossary": glossary, "multi_hop": multi},
        cfg, rng,
    )

    counts = Counter(p["meta"]["kind"].split("_")[0] for p in final)
    log.info("Dataset final: %d exemplos | mix=%s", len(final), dict(counts))

    out_path = Path(cfg["output_path"])
    n = write_jsonl(final, out_path)
    log.info("Salvo em %s (%d linhas)", out_path, n)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(ap.parse_args().config)


if __name__ == "__main__":
    main()
