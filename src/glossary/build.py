"""Constrói o glossário completo: entradas curadas + auto-extraídas.

- Curadas: data/glossary/glossary.yaml (definições técnicas em PT escritas
  manualmente para radar, lidar, termodinâmica, radiação, aerossol,
  instrumentos e temas ARM).
- Auto: variáveis presentes nos JSON-LD que NÃO constam no glossário curado
  ganham uma entrada automática usando a descrição mais frequente encontrada
  nos próprios metadados (em inglês, marcada como source=auto).

Saída: data/glossary/glossary_full.json — consumido pelo gerador de QA.

Uso:
    python -m src.glossary.build --json-dir data/raw_json \
        --curated data/glossary/glossary.yaml \
        --out data/glossary/glossary_full.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.io import load_json, load_yaml, save_json, setup_logger

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


def harvest_auto(json_dir: Path, covered: set[str],
                 min_count: int, max_auto: int) -> list[dict[str, Any]]:
    """Extrai definições automáticas das descrições de variáveis dos JSONs."""
    # var_lower -> Counter de descrições; e Counter de grafias
    desc_by_var: dict[str, Counter] = defaultdict(Counter)
    casing_by_var: dict[str, Counter] = defaultdict(Counter)
    n_datasets_by_var: Counter = Counter()

    files = sorted(json_dir.glob("*.json"))
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

    candidates = []
    for key, n_ds in n_datasets_by_var.items():
        if key in covered or n_ds < min_count:
            continue
        descs = desc_by_var.get(key)
        if not descs:
            continue
        best_desc, _ = descs.most_common(1)[0]
        best_name, _ = casing_by_var[key].most_common(1)[0]
        aliases = [c for c in casing_by_var[key] if c != best_name]
        candidates.append({
            "term": best_name,
            "aliases": aliases,
            "category": "auto",
            "unit": None,
            "definition": best_desc,   # em inglês, vindo dos metadados
            "uso": None,
            "expansao": None,
            "source": "auto",
            "n_datasets": n_ds,
        })

    candidates.sort(key=lambda e: -e["n_datasets"])
    log.info("Variáveis sem entrada curada com descrição: %d "
             "(mantendo top %d)", len(candidates), max_auto)
    return candidates[:max_auto]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--curated", default="data/glossary/glossary.yaml")
    ap.add_argument("--out", default="data/glossary/glossary_full.json")
    ap.add_argument("--min-count", type=int, default=2,
                    help="mín. de datasets em que a variável aparece")
    ap.add_argument("--max-auto", type=int, default=400,
                    help="máx. de entradas automáticas")
    args = ap.parse_args()

    curated = load_curated(Path(args.curated))
    covered = alias_index(curated)
    auto = harvest_auto(Path(args.json_dir), covered,
                        args.min_count, args.max_auto)

    out = {
        "n_curated": len(curated),
        "n_auto": len(auto),
        "entries": curated + auto,
    }
    save_json(out, args.out)
    log.info("Glossário salvo em %s (curadas=%d, auto=%d, total=%d)",
             args.out, len(curated), len(auto), len(curated) + len(auto))


if __name__ == "__main__":
    main()
