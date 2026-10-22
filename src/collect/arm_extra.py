"""Coleta metadados EXTRAS de serviços oficiais da ARM e monta um catálogo.

Parte do corpus que já temos (JSON-LD da Data Discovery em data/raw_json):
de cada registro tiramos a CLASSE do produto (ex.: "1twrmr"), o SÍTIO e a
FACILITY (da citação: "Southern Great Plains (SGP) ... (C1)") e o período.
Com isso consultamos (veja src/collect/arm_services.py):

  1. DOD   -> níveis de processamento existentes (a1, b1, c1...) e, para cada
              um, as variáveis com nome longo e UNIDADE;
  2. DQR   -> relatórios de qualidade de cada datastream (ex.: sgp1twrmrC1.c1);
  3. sítios-> facilities de cada sítio com coordenadas;
  4. (opcional) citação oficial com DOI de cada datastream.

Saída: data/arm_extra/catalog.json (+ respostas brutas em cache), consumido por
    python -m src.glossary.build --extra ...   (unidades no glossário)
    python -m src.rag.build_index              (unidades, níveis e DQRs no RAG)

Uso:
    python -m src.collect.arm_extra --dry-run            # só mostra o plano
    python -m src.collect.arm_extra --max-classes 20     # teste rápido
    python -m src.collect.arm_extra                      # tudo (com cache)
    python -m src.collect.arm_extra --datastreams lista.txt   # nomes completos
    python -m src.collect.arm_extra --explorar-site mao  # inspeciona o índice

Os serviços são públicos (sem login). O cache em data/arm_extra/cache evita
repetir requisições: rodar de novo só busca o que falta.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from src.common.arm_names import parse_datastream
from src.common.io import (list_json_files, load_json, load_yaml, save_json,
                           setup_logger)
from src.collect.arm_services import (
    DEFAULT_USER_AGENT, DOD_URL, DQR_URL, CITATION_URL, METADATA_SEARCH_URL,
    ArmClient, ArmServiceError, doi_from_text, fetch_citation, fetch_dod,
    fetch_dqrs, fetch_facilities, parse_dod, parse_dqrs,
)
from src.prompts.generator import parse_record

log = setup_logger("collect.arm_extra")

CATALOG_VERSION = 1
# Ordem de tentativa dos níveis quando o nível do produto não é conhecido.
DEFAULT_LEVELS = ["b1", "c1", "a1", "c2", "a0", "s1"]
CLASS_RE = re.compile(r"^[a-z0-9]+$")
SITE_RE = re.compile(r"\(([A-Za-z]{3})\)")
FACILITY_RE = re.compile(r"\(([A-Z][0-9]{1,3})\)")


def _yyyymmdd(s: str | None, default: str) -> str:
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", s or "")
    return "".join(m.groups()) if m else default


def _iso(s: str | None, default: str) -> str:
    m = re.match(r"^\s*(\d{4}-\d{2}-\d{2})", s or "")
    return m.group(1) if m else default


# --------------------------------------------------------------------------- #
# O que já sabemos a partir do corpus
# --------------------------------------------------------------------------- #
def corpus_targets(json_dir: str | Path) -> dict[str, dict[str, Any]]:
    """classe -> {sites: {(site, facility)}, start, end, names: [...]}"""
    targets: dict[str, dict[str, Any]] = {}
    files = list_json_files(json_dir)
    for p in files:
        try:
            rec = parse_record(load_json(p))
        except Exception:
            continue
        if not rec:
            continue
        code = (rec.short_code or "").strip().lower()
        cls, level = (code.split(".", 1) + [None])[:2] if "." in code else (code, None)
        if not CLASS_RE.match(cls or ""):
            continue
        t = targets.setdefault(cls, {"pairs": set(), "levels": set(),
                                     "start": None, "end": None, "names": []})
        t["names"].append(rec.name)
        if level:
            t["levels"].add(level)
        detail = rec.site_detail or ""
        site = SITE_RE.search(detail)
        fac = FACILITY_RE.findall(detail)
        if site and fac:
            t["pairs"].add((site.group(1).lower(), fac[-1]))
        if rec.period_start:
            s = _iso(rec.period_start, "")
            if s and (t["start"] is None or s < t["start"]):
                t["start"] = s
        end = None if rec.ongoing else _iso(rec.period_end, "")
        if rec.ongoing:
            t["end"] = "ongoing"
        elif end and t["end"] != "ongoing" and (t["end"] is None or end > t["end"]):
            t["end"] = end
    log.info("Corpus: %d arquivos -> %d classes de produto (%d com sítio/"
             "facility identificados)", len(files), len(targets),
             sum(1 for t in targets.values() if t["pairs"]))
    return targets


def read_datastream_list(path: str | Path) -> list[dict[str, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        ds = parse_datastream(line)
        if ds:
            out.append(ds)
        else:
            log.warning("Nome de datastream inválido, ignorado: %r", line)
    return out


# --------------------------------------------------------------------------- #
# Coleta
# --------------------------------------------------------------------------- #
class ArmUnreachable(RuntimeError):
    """Muitas consultas seguidas falharam: rede bloqueada ou serviço fora do ar."""


class FailureGuard:
    """Interrompe a coleta depois de N erros SEGUIDOS (404 não conta: é só
    "não existe"). Evita horas de tentativas inúteis sem rede."""

    def __init__(self, limit: int = 10):
        self.limit, self.n = limit, 0

    def ok(self) -> None:
        self.n = 0

    def fail(self, err: Exception) -> None:
        self.n += 1
        if self.limit and self.n >= self.limit:
            raise ArmUnreachable(
                f"{self.n} consultas seguidas falharam (última: {err}). A rede "
                "ou os serviços da ARM parecem inacessíveis daqui; o que já "
                "veio está no cache — rode de novo mais tarde.")


class Progress:
    def __init__(self, total: int, label: str):
        self.total, self.label, self.done, self.t0 = total, label, 0, time.time()

    def step(self) -> None:
        self.done += 1
        if self.done % 25 == 0 or self.done == self.total:
            el = time.time() - self.t0
            eta = el / self.done * (self.total - self.done)
            log.info("[%s] %d/%d | %.0fs decorridos | ETA %.0fs",
                     self.label, self.done, self.total, el, eta)


def collect(targets: dict[str, dict], explicit: list[dict], client: ArmClient,
            levels: list[str], do_dods: bool, do_dqrs: bool,
            do_citations: bool, do_sites: bool,
            max_classes: int | None = None,
            max_consecutive_errors: int = 10) -> dict[str, Any]:
    today = date.today()
    guard = FailureGuard(max_consecutive_errors)
    classes: dict[str, dict[str, Any]] = {}
    for ds in explicit:
        t = targets.setdefault(ds["class"], {"pairs": set(), "levels": set(),
                                             "start": None, "end": None,
                                             "names": []})
        t["levels"].add(ds["level"])
        t["pairs"].add((ds["site"], ds["facility"]))
    names = sorted(targets)
    if max_classes:
        names = names[:max_classes]

    # 1) DODs: níveis existentes + variáveis
    if do_dods:
        prog = Progress(sum(len(targets[c]["levels"]) or len(levels)
                            for c in names), "DOD")
        for cls in names:
            t = targets[cls]
            entry = classes.setdefault(cls, {"levels": {}, "datastreams": []})
            for lvl in (sorted(t["levels"]) or levels):
                proc = f"{cls}.{lvl}"
                try:
                    dod = parse_dod(fetch_dod(client, proc), proc)
                    guard.ok()
                except ArmServiceError as e:
                    log.warning("DOD %s: %s", proc, e)
                    guard.fail(e)
                    dod = None
                prog.step()
                if dod:
                    entry["levels"][lvl] = dod
    else:
        for cls in names:
            classes.setdefault(cls, {"levels": {}, "datastreams": []})

    # 2) datastreams (site + classe + facility + nível) e DQRs/citações.
    # Os nomes são MONTADOS a partir do corpus + DOD, e nem toda combinação
    # existe: só marcamos "confirmed" com evidência (DQR ou citação devolvidos,
    # ou nome informado explicitamente em --datastreams).
    explicit_names = {d["datastream"] for d in explicit}
    all_dqrs: list[dict] = []
    plan = []
    for cls in names:
        t = targets[cls]
        found_levels = sorted(classes[cls]["levels"]) or sorted(t["levels"])
        for site, fac in sorted(t["pairs"]):
            for lvl in found_levels:
                plan.append((cls, f"{site}{cls}{fac}.{lvl}", t))
    if do_dqrs or do_citations:
        prog = Progress(len(plan) * (int(do_dqrs) + int(do_citations)),
                        "DQR/citação")
        for cls, ds_name, t in plan:
            info = parse_datastream(ds_name) or {"datastream": ds_name}
            entry = {k: info.get(k) for k in ("datastream", "site", "facility",
                                              "level")}
            entry["confirmed"] = ds_name in explicit_names
            start = t["start"] or "1993-01-01"
            end = today.isoformat() if t["end"] in (None, "ongoing") else t["end"]
            if do_dqrs:
                try:
                    dqrs = parse_dqrs(fetch_dqrs(
                        client, ds_name, _yyyymmdd(start, "19930101"),
                        _yyyymmdd(end, today.strftime("%Y%m%d"))), ds_name)
                    guard.ok()
                except ArmServiceError as e:
                    log.warning("DQR %s: %s", ds_name, e)
                    guard.fail(e)
                    dqrs = []
                prog.step()
                for d in dqrs:
                    d["class"] = cls
                all_dqrs.extend(dqrs)
                entry["dqr_counts"] = dict(Counter(d["category"] for d in dqrs))
                entry["confirmed"] = entry["confirmed"] or bool(dqrs)
            if do_citations:
                try:
                    cit = fetch_citation(client, ds_name, _iso(start, start),
                                         _iso(end, end))
                    guard.ok()
                except ArmServiceError as e:
                    log.warning("Citação %s: %s", ds_name, e)
                    guard.fail(e)
                    cit = None
                prog.step()
                entry["citation"] = cit
                entry["doi"] = doi_from_text(cit)
                entry["confirmed"] = entry["confirmed"] or bool(cit)
            classes[cls]["datastreams"].append(entry)

    # 3) sítios/facilities com coordenadas
    sites: list[dict] = []
    if do_sites:
        site_codes = sorted({s for c in names for s, _ in targets[c]["pairs"]})
        for sc in site_codes:
            try:
                sites.extend(fetch_facilities(client, sc))
                guard.ok()
            except ArmServiceError as e:
                log.warning("Sítio %s: %s", sc, e)
                guard.fail(e)

    return {"classes": classes, "dqrs": all_dqrs, "sites": sites}


def build_catalog(result: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    classes = result["classes"]
    n_levels = sum(len(c["levels"]) for c in classes.values())
    n_vars = sum(len(lv["variables"]) for c in classes.values()
                 for lv in c["levels"].values())
    n_units = sum(1 for c in classes.values() for lv in c["levels"].values()
                  for v in lv["variables"] if v.get("units"))
    return {
        "version": CATALOG_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {
            "dod": DOD_URL, "dqr": DQR_URL, "citation": CITATION_URL,
            "metadata_search": METADATA_SEARCH_URL,
            "reference": "formatos verificados em ACT 2.3.4 "
                         "(github.com/ARM-DOE/ACT)",
        },
        "settings": args,
        "stats": {
            "classes": len(classes),
            "classes_with_dod": sum(1 for c in classes.values() if c["levels"]),
            "levels": n_levels,
            "variables": n_vars,
            "variables_with_units": n_units,
            "datastreams": sum(len(c["datastreams"]) for c in classes.values()),
            "datastreams_confirmed": sum(
                1 for c in classes.values() for d in c["datastreams"]
                if d.get("confirmed")),
            "dqrs": len(result["dqrs"]),
            "dqrs_by_category": dict(Counter(d["category"]
                                             for d in result["dqrs"])),
            "facilities": len(result["sites"]),
        },
        "classes": classes,
        "dqrs": result["dqrs"],
        "sites": result["sites"],
    }


# --------------------------------------------------------------------------- #
# Exploração do índice de metadados (campos não documentados publicamente)
# --------------------------------------------------------------------------- #
def explore_site(client: ArmClient, site_code: str, out_dir: Path,
                 size: int = 20) -> Path:
    """Baixa alguns documentos brutos do índice da Data Discovery e lista os
    campos existentes — para decidir o que mais vale extrair dele."""
    body = {"size": size, "query": {"query_string": {
        "query": f"site_code:{site_code}"}}}
    raw = client.post_json(METADATA_SEARCH_URL, body) or {}
    hits = ((raw.get("hits") or {}).get("hits")) or []
    fields: Counter = Counter()
    examples: dict[str, Any] = {}
    for h in hits:
        for k, v in (h.get("_source") or {}).items():
            fields[k] += 1
            examples.setdefault(k, v)
    out = out_dir / f"explorar_{site_code}.json"
    save_json({"site_code": site_code, "n_hits": len(hits),
               "total": (raw.get("hits") or {}).get("total"),
               "fields": dict(fields.most_common()),
               "examples": examples, "raw_hits": hits}, out)
    print(f"\n{len(hits)} documentos do sítio {site_code}; campos encontrados:")
    for k, n in fields.most_common():
        ex = repr(examples[k])
        print(f"  {k:<32} {n:>3}x   ex.: {ex[:70]}")
    print(f"\nDocumentos brutos salvos em {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/collect.yaml",
                    help="usa user_agent e request_delay_seconds de lá")
    ap.add_argument("--json-dir", default="data/raw_json")
    ap.add_argument("--datastreams", default=None,
                    help="arquivo com nomes completos (ex.: maoarmbeatmM1.c1), "
                         "um por linha")
    ap.add_argument("--out", default="data/arm_extra/catalog.json")
    ap.add_argument("--cache", default="data/arm_extra/cache")
    ap.add_argument("--niveis", default=",".join(DEFAULT_LEVELS),
                    help="níveis a testar quando o nível do produto não é "
                         "conhecido")
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--sem-dods", action="store_true")
    ap.add_argument("--sem-dqrs", action="store_true")
    ap.add_argument("--sem-sitios", action="store_true")
    ap.add_argument("--citacoes", action="store_true",
                    help="também busca a citação oficial de cada datastream")
    ap.add_argument("--atualizar", action="store_true",
                    help="ignora o cache e consulta tudo de novo")
    ap.add_argument("--dry-run", action="store_true",
                    help="só mostra quantas consultas seriam feitas")
    ap.add_argument("--explorar-site", default=None, metavar="SITE",
                    help="lista os campos do índice de metadados para um "
                         "sítio (ex.: mao) e sai")
    a = ap.parse_args()

    ua, delay = DEFAULT_USER_AGENT, 0.5
    if a.config and Path(a.config).exists():
        src = ((load_yaml(a.config) or {}).get("sources") or [{}])[0]
        ua = src.get("user_agent", ua)
        delay = float(src.get("request_delay_seconds", delay))
    client = ArmClient(cache_dir=a.cache, user_agent=ua, delay_s=delay,
                       refresh=a.atualizar)

    if a.explorar_site:
        explore_site(client, a.explorar_site.lower(), Path(a.out).parent)
        return

    targets = corpus_targets(a.json_dir)
    explicit = read_datastream_list(a.datastreams) if a.datastreams else []
    if not targets and not explicit:
        # não sobrescreve um catálogo bom com um vazio
        log.error("Nenhum produto encontrado: %s não tem JSON-LD (pasta "
                  "atual: %s). Confira se está na raiz do projeto e se os "
                  "JSONs estão em data/raw_json. Nada foi salvo.",
                  Path(a.json_dir).resolve(), Path.cwd())
        sys.exit(2)
    levels = [x.strip() for x in a.niveis.split(",") if x.strip()]

    if a.dry_run:
        names = sorted(set(targets) | {d["class"] for d in explicit})
        if a.max_classes:
            names = names[: a.max_classes]
        n_dod = sum(len(targets.get(c, {}).get("levels", ())) or len(levels)
                    for c in names)
        n_pairs = sum(len(targets.get(c, {}).get("pairs", ())) for c in names)
        print(f"Classes: {len(names)} | consultas DOD (máx.): {n_dod} | "
              f"pares sítio/facility: {n_pairs} | datastreams explícitos: "
              f"{len(explicit)}")
        print(f"Com {delay}s entre consultas, os DODs levam até "
              f"~{n_dod * delay / 60:.0f} min na 1ª vez (depois vem do cache).")
        return

    t0 = time.time()
    try:
        result = collect(targets, explicit, client, levels,
                         do_dods=not a.sem_dods, do_dqrs=not a.sem_dqrs,
                         do_citations=a.citacoes, do_sites=not a.sem_sitios,
                         max_classes=a.max_classes)
    except ArmUnreachable as e:
        # não sobrescreve um catálogo anterior com um vazio
        log.error("%s", e)
        sys.exit(2)
    catalog = build_catalog(result, {
        "json_dir": a.json_dir, "datastreams_file": a.datastreams,
        "levels": levels, "max_classes": a.max_classes,
        "citations": a.citacoes})
    save_json(catalog, a.out)
    s = catalog["stats"]
    log.info("Catálogo salvo em %s (%.0fs; %d consultas, %d do cache)",
             a.out, time.time() - t0, client.n_requests, client.n_cache_hits)
    log.info("Classes com DOD: %d/%d | níveis: %d | variáveis: %d (%d com "
             "unidade) | datastreams: %d | DQRs: %d %s | facilities: %d",
             s["classes_with_dod"], s["classes"], s["levels"], s["variables"],
             s["variables_with_units"], s["datastreams"], s["dqrs"],
             s["dqrs_by_category"], s["facilities"])
    if not a.sem_dods:
        com = [c for c, v in catalog["classes"].items() if v["levels"]]
        sem = [c for c, v in catalog["classes"].items() if not v["levels"]]
        log.info("Com DOD (%d): %s", len(com), ", ".join(
            f"{c} [{','.join(sorted(catalog['classes'][c]['levels']))}]"
            for c in com[:30]) or "-")
        if sem:
            log.info("Sem DOD nos níveis %s (%d): %s%s", ",".join(levels),
                     len(sem), ", ".join(sem[:40]),
                     " ..." if len(sem) > 40 else "")


if __name__ == "__main__":
    main()
