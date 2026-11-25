"""Constrói o índice RAG a partir dos JSON-LD + glossário.

Uso (da raiz do projeto, NÃO precisa de GPU — leva segundos):
    python -m assistente_metadados.rag.build_index
    python -m assistente_metadados.rag.build_index --json-dir data/raw_json \
        --glossary data/glossary/glossary_full.json --out data/rag/index.json

Sem --glossary, usa data/glossary/glossary_full.json (curado + auto) se existir;
senão cai para o curado data/glossary/glossary.yaml.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from assistente_metadados.common.arm_catalog import DEFAULT_CATALOG, load_catalog
from assistente_metadados.common.io import list_json_files, load_json, load_yaml, setup_logger
from assistente_metadados.prompts.generator import Record, parse_record
from assistente_metadados.rag.documents import build_documents
from assistente_metadados.rag.extra import apply_extra
from assistente_metadados.rag.index import Index, load_expansions

log = setup_logger("rag.build_index")

DEFAULT_GLOSSARY_FULL = "data/glossary/glossary_full.json"
DEFAULT_GLOSSARY_CURATED = "data/glossary/glossary.yaml"
DEFAULT_EXPANSIONS = "data/rag/expansoes_pt_en.yaml"
DEFAULT_EXTRA = DEFAULT_CATALOG
DEFAULT_OUT = "data/rag/index.json"


def load_records(json_dir: str | Path) -> list[Record]:
    files = list_json_files(json_dir)
    records: list[Record] = []
    for p in files:
        try:
            rec = parse_record(load_json(p))
            if rec:
                records.append(rec)
        except Exception as e:  # um JSON ruim não derruba o índice
            log.warning("Falha em %s: %s", p.name, e)
    log.info("JSONs lidos: %d | registros válidos: %d", len(files), len(records))
    return records


def load_glossary(path: str | Path | None) -> tuple[list[dict[str, Any]], str | None]:
    candidates = [path] if path else [DEFAULT_GLOSSARY_FULL,
                                      DEFAULT_GLOSSARY_CURATED]
    for c in candidates:
        if c and Path(c).exists():
            if str(c).endswith((".yaml", ".yml")):
                entries = (load_yaml(c) or {}).get("entries", [])
                for e in entries:
                    e.setdefault("source", "curated")
            else:
                entries = load_json(c).get("entries", [])
            return entries, str(c)
    return [], None


def load_extra(path: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Catálogo de serviços ARM (assistente_metadados.collect.arm_extra). "auto" = usa o
    caminho padrão se existir; "none" = não usa."""
    data, used = load_catalog(path)
    if data is None and path and str(path).lower() not in (
            "auto", "none", "nenhum", "0"):
        log.warning("Catálogo extra %s não encontrado; seguindo sem ele.", path)
    return data, used


def build(json_dir: str, glossary: str | None, expansions: str | None,
          out: str, extra: str | None = "auto") -> Index:
    t0 = time.time()
    records = load_records(json_dir)
    if not records:
        log.warning("Nenhum JSON válido em %s. Você está na raiz do projeto "
                    "e copiou os JSON-LD para data/raw_json? (veja o README, "
                    "passo 3.1)", json_dir)
    entries, gpath = load_glossary(glossary)
    if gpath:
        log.info("Glossário: %s (%d entradas)", gpath, len(entries))
    else:
        log.warning("Glossário não encontrado; o índice ficará sem as "
                    "definições. Rode assistente_metadados.glossary.build antes.")
    exp = load_expansions(expansions)
    log.info("Expansões PT->EN: %d termos", len(exp))

    docs = build_documents(records, entries)
    extra_data, extra_path = load_extra(extra)
    if extra_data:
        docs = apply_extra(docs, extra_data)
        log.info("Catálogo extra da ARM: %s | %s", extra_path,
                 extra_data.get("stats"))
    index = Index(docs, expansions=exp, info={
        "json_dir": str(json_dir), "n_records": len(records),
        "glossary": gpath, "expansions": expansions,
        "extra": extra_path,
        "extra_built_at": (extra_data or {}).get("built_at"),
    })
    index.save(out)
    kinds: dict[str, int] = {}
    for d in docs:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
    size_mb = Path(out).stat().st_size / 1e6
    log.info("Índice salvo em %s (%.1f MB, %.1fs) | documentos: %d %s",
             out, size_mb, time.time() - t0, len(docs), kinds)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default="data/raw_json")
    ap.add_argument("--glossary", default=None,
                    help="glossary_full.json ou glossary.yaml (padrão: automático)")
    ap.add_argument("--expansions", default=DEFAULT_EXPANSIONS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--extra", default="auto",
                    help="catálogo de assistente_metadados.collect.arm_extra (DOD/DQR/sítios). "
                         f"'auto' = {DEFAULT_EXTRA} se existir; 'none' = sem")
    a = ap.parse_args()
    build(a.json_dir, a.glossary, a.expansions, a.out, extra=a.extra)


if __name__ == "__main__":
    main()
