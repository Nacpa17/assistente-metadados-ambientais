"""Constrói o glossário completo: entradas curadas + auto-extraídas.

- Curadas: data/glossary/glossary.yaml (definições técnicas em PT escritas
  manualmente para radar, lidar, termodinâmica, radiação, aerossol,
  instrumentos e temas ARM).
- Auto: variáveis presentes nos JSON-LD que NÃO constam no glossário curado
  ganham uma entrada automática usando a descrição mais frequente encontrada
  nos próprios metadados (em inglês, marcada como source=auto).
- DOD (opcional, --extra): com o catálogo de serviços oficiais da ARM
  (`python -m assistente_metadados.collect.arm_extra`), as entradas automáticas ganham a
  UNIDADE oficial declarada nos DODs (unit_source=dod) e variáveis que só
  existem nos DODs viram entradas novas (source=dod, definição = long_name
  oficial). Variáveis de controle de qualidade (qc_*/aqc_*) ficam de fora.
  As entradas curadas nunca são alteradas.

Saída: data/glossary/glossary_full.json — consumido pelo gerador de QA e
pelo índice RAG.

Uso:
    python -m assistente_metadados.glossary.build --json-dir data/raw_json \
        --curated data/glossary/glossary.yaml \
        --out data/glossary/glossary_full.json
    # com unidades/nomes longos oficiais (catálogo da ARM):
    python -m assistente_metadados.glossary.build --json-dir data/raw_json --extra auto

Sem --extra as entradas são idênticas às da fase 2 (mantém o baseline
reprodutível); use --extra ao montar o dataset do novo fine-tuning.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from assistente_metadados.common.arm_catalog import load_catalog, unit_split, variable_info
from assistente_metadados.common.io import (list_json_files, load_json, load_yaml, save_json,
                           setup_logger)

log = setup_logger("glossary.build")


def load_curated(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    entries = data.get("entries", [])
    for e in entries:
        e.setdefault("aliases", [])
        e["source"] = "curated"
    return entries


def alias_index(entries: list[dict]) -> set[str]:
    """Todos os termos+aliases curados, normalizados p/ lowercase."""
    idx: set[str] = set()
    for e in entries:
        idx.add(e["term"].lower())
        for a in e.get("aliases", []):
            idx.add(str(a).lower())
    return idx


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.lower()).strip()


def useful_long_name(name: str, long_name: str | None) -> bool:
    """Descarta nome longo vazio ou que só repete o nome ("time" -> "time")."""
    return bool(long_name) and _norm(long_name) != _norm(name)


def add_dod_fields(entry: dict[str, Any], d: dict[str, Any]) -> bool:
    """Acrescenta unidade/standard_name oficiais do DOD. True se pôs unidade."""
    entry["n_products_dod"] = len(d["classes"])
    if d.get("standard_name"):
        entry["standard_name"] = d["standard_name"]
    top, alts = unit_split(d["units"])
    if top and not entry.get("unit"):
        entry["unit"] = top
        entry["unit_alternatives"] = alts
        entry["unit_source"] = "dod"
        return True
    return False


def scan_corpus(json_dir: Path) -> tuple[dict[str, Counter], dict[str, Counter],
                                         Counter]:
    """(descrições por variável, grafias por variável, nº de datasets)."""
    desc_by_var: dict[str, Counter] = defaultdict(Counter)
    casing_by_var: dict[str, Counter] = defaultdict(Counter)
    n_datasets_by_var: Counter = Counter()

    files = list_json_files(json_dir)
    log.info("Varredura de %d JSONs para definições automáticas", len(files))
    for p in files:
        try:
            d = load_json(p)
        except Exception:
            continue
        seen_in_this: set[str] = set()
        for v in d.get("variableMeasured") or []:
            if not isinstance(v, dict):
                continue
            name = (v.get("name") or "").strip()
            desc = (v.get("description") or "").strip()
            if not name:
                continue
            key = name.lower()
            casing_by_var[key][name] += 1
            if desc:
                desc_by_var[key][desc] += 1
            if key not in seen_in_this:
                n_datasets_by_var[key] += 1
                seen_in_this.add(key)
    return desc_by_var, casing_by_var, n_datasets_by_var


def harvest_auto(json_dir: Path, covered: set[str], min_count: int,
                 max_auto: int, dod: dict[str, dict] | None = None,
                 scan: tuple | None = None) -> list[dict[str, Any]]:
    """Extrai definições automáticas das descrições de variáveis dos JSONs.

    Com `dod` (variable_info do catálogo): unidade oficial nas entradas e,
    quando o JSON-LD não traz descrição, o long_name do DOD (source=dod).
    """
    desc_by_var, casing_by_var, n_datasets_by_var = scan or scan_corpus(json_dir)
    dod = dod or {}

    candidates = []
    for key, n_ds in n_datasets_by_var.items():
        if key in covered or n_ds < min_count:
            continue
        best_name, _ = casing_by_var[key].most_common(1)[0]
        d = dod.get(key)
        descs = desc_by_var.get(key)
        if descs:
            definition, source = descs.most_common(1)[0][0], "auto"
        elif d and useful_long_name(best_name, d.get("long_name")):
            definition, source = d["long_name"], "dod"
        else:
            continue
        aliases = [c for c in casing_by_var[key] if c != best_name]
        entry = {
            "term": best_name,
            "aliases": aliases,
            "category": "auto",
            "unit": None,
            "definition": definition,   # em inglês, vindo dos metadados
            "uso": None,
            "expansao": None,
            "source": source,
            "n_datasets": n_ds,
        }
        if d:
            add_dod_fields(entry, d)
        candidates.append(entry)

    candidates.sort(key=lambda e: -e["n_datasets"])
    log.info("Variáveis sem entrada curada com descrição: %d "
             "(mantendo top %d)", len(candidates), max_auto)
    return candidates[:max_auto]


def harvest_dod(dod: dict[str, dict], covered: set[str], min_products: int,
                max_dod: int) -> list[dict[str, Any]]:
    """Variáveis que só existem nos DODs oficiais (fora do corpus JSON-LD)."""
    out = []
    for key, d in dod.items():
        if key in covered or len(d["classes"]) < min_products:
            continue
        if not useful_long_name(d["name"], d.get("long_name")):
            continue
        entry = {
            "term": d["name"],
            "aliases": [],
            "category": "dod",
            "unit": None,
            "definition": d["long_name"],   # long_name oficial (inglês)
            "uso": None,
            "expansao": None,
            "source": "dod",
            "n_datasets": 0,
        }
        add_dod_fields(entry, d)
        out.append(entry)
    # mais produtos primeiro; com unidade antes de sem; depois alfabética
    out.sort(key=lambda e: (-e["n_products_dod"], e.get("unit") is None,
                            e["term"].lower()))
    log.info("Variáveis só nos DODs: %d (mantendo top %d)", len(out), max_dod)
    return out[:max_dod]


def build_glossary(json_dir: Path, curated_path: Path, min_count: int = 2,
                   max_auto: int = 400, extra: str | None = None,
                   max_dod: int = 300, min_products: int = 1) -> dict[str, Any]:
    curated = load_curated(curated_path)
    covered = alias_index(curated)
    catalog, catalog_path = load_catalog(extra)
    if extra and str(extra).lower() not in ("none", "nenhum", "0") \
            and catalog is None:
        log.warning("Catálogo %s não encontrado (rode assistente_metadados.collect.arm_extra); "
                    "seguindo sem DOD.", extra)
    dod = variable_info(catalog) if catalog else {}

    scan = scan_corpus(json_dir)
    auto = harvest_auto(json_dir, covered, min_count, max_auto, dod=dod,
                        scan=scan)
    dod_entries: list[dict[str, Any]] = []
    if dod:
        # "só no DOD" = nem curado nem presente em QUALQUER JSON-LD do corpus
        in_corpus = set(scan[2])
        dod_entries = harvest_dod(dod, covered | in_corpus, min_products,
                                  max_dod)
    entries = curated + auto + dod_entries
    out: dict[str, Any] = {
        "n_curated": len(curated),
        "n_auto": len(auto),
        "n_dod": len(dod_entries),
        "entries": entries,
    }
    if catalog:
        out["extra"] = {
            "catalog": catalog_path,
            "catalog_built_at": catalog.get("built_at"),
            "n_dod_variables": len(dod),
            "n_units_from_dod": sum(1 for e in entries
                                    if e.get("unit_source") == "dod"),
            "n_definitions_from_dod": sum(1 for e in auto
                                          if e["source"] == "dod")
                                      + len(dod_entries),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--curated", default="data/glossary/glossary.yaml")
    ap.add_argument("--out", default="data/glossary/glossary_full.json")
    ap.add_argument("--min-count", type=int, default=2,
                    help="mín. de datasets em que a variável aparece")
    ap.add_argument("--max-auto", type=int, default=400,
                    help="máx. de entradas automáticas")
    ap.add_argument("--extra", default="none",
                    help="catálogo de assistente_metadados.collect.arm_extra para unidades e "
                         "nomes longos oficiais (DOD). 'auto' = "
                         "data/arm_extra/catalog.json; padrão: none")
    ap.add_argument("--max-dod", type=int, default=300,
                    help="máx. de entradas novas vindas só dos DODs")
    ap.add_argument("--min-produtos", dest="min_products", type=int, default=1,
                    help="mín. de produtos (classes) cujo DOD tem a variável")
    args = ap.parse_args()

    out = build_glossary(Path(args.json_dir), Path(args.curated),
                         args.min_count, args.max_auto, extra=args.extra,
                         max_dod=args.max_dod, min_products=args.min_products)
    save_json(out, args.out)
    log.info("Glossário salvo em %s (curadas=%d, auto=%d, dod=%d, total=%d)",
             args.out, out["n_curated"], out["n_auto"], out["n_dod"],
             len(out["entries"]))
    if out.get("extra"):
        log.info("DOD: %s", out["extra"])


if __name__ == "__main__":
    main()
