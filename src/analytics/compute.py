"""Pré-computa estatísticas agregadas sobre o corpus de JSON-LD.

v3 — correções:
- temporalCoverage terminando em ".." = coleta contínua (usa ano atual)
- anos placeholder (> 2100, ex.: 2999) são ignorados
- novo campo: ongoing_datasets (contagem de séries ativas)
- novo campo: facilities (contagem por observatório, extraída de `about`)

Uso:
    python -m src.analytics.compute --in data/raw_json --out data/analytics/stats.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from src.common.io import load_json, save_json, setup_logger

log = setup_logger("analytics.compute")

MAX_SANE_YEAR = 2100


def safe_year(date_str: str) -> int | None:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y"):
        try:
            y = datetime.strptime(date_str.strip(), fmt).year
            return y if 1900 <= y <= MAX_SANE_YEAR else None
        except ValueError:
            continue
    return None


def parse_temporal(d: dict[str, Any]) -> tuple[int | None, int | None, bool]:
    """Retorna (ano_inicio, ano_fim, ongoing)."""
    t = d.get("temporalCoverage")
    if not isinstance(t, str) or "/" not in t:
        return None, None, False
    a, b = (s.strip() for s in t.split("/", 1))
    ya = safe_year(a)
    if b in ("..", ""):
        return ya, None, True
    yb = safe_year(b)
    if yb is None:
        return ya, None, True
    return ya, yb, False


def iter_temporal_years(d: dict[str, Any]):
    ya, yb, ongoing = parse_temporal(d)
    if ya is None:
        return
    end = datetime.now().year if ongoing else yb
    if end is None or end < ya:
        return
    for y in range(ya, end + 1):
        yield y


def extract_variables(d: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for v in d.get("variableMeasured") or []:
        if isinstance(v, dict) and v.get("name"):
            out.append(v["name"].strip())
    return out


def extract_keywords(d: dict[str, Any]) -> list[str]:
    return [k.strip() for k in (d.get("keywords") or []) if isinstance(k, str)]


def extract_org_name(d: dict[str, Any], field: str) -> str | None:
    v = d.get(field)
    if isinstance(v, dict):
        name = v.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def extract_facility(d: dict[str, Any]) -> str | None:
    for entry in d.get("about") or []:
        if isinstance(entry, dict):
            n = (entry.get("name") or "").strip()
            if n.startswith("Facility Information:"):
                return n.split(":", 1)[1].strip()
    return None


def extract_lat_lon(d: dict[str, Any]) -> tuple[float, float] | None:
    geo = d.get("spatialCoverage")
    if isinstance(geo, list) and geo and isinstance(geo[0], dict):
        g = geo[0].get("geo") or {}
        lat, lon = g.get("latitude"), g.get("longitude")
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except (TypeError, ValueError):
                return None
    return None


def compute(input_dir: Path, top_k: int = 20) -> dict[str, Any]:
    files = sorted(input_dir.glob("*.json"))
    log.info("Analisando %d arquivos em %s", len(files), input_dir)

    var_counter: Counter[str] = Counter()
    kw_counter: Counter[str] = Counter()
    provider_counter: Counter[str] = Counter()
    publisher_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    year_counter: Counter[int] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()
    facility_counter: Counter[str] = Counter()

    lats: list[float] = []
    lons: list[float] = []
    total = 0
    ongoing_total = 0

    for p in files:
        try:
            d = load_json(p)
        except Exception as e:
            log.warning("Falha lendo %s: %s", p, e)
            continue
        if not isinstance(d, dict):
            continue
        total += 1

        vars_ = extract_variables(d)
        var_counter.update(vars_)
        for a, b in combinations(sorted(set(vars_)), 2):
            pair_counter[(a, b)] += 1

        kw_counter.update(extract_keywords(d))

        if (org := extract_org_name(d, "provider")):
            provider_counter[org] += 1
        if (org := extract_org_name(d, "publisher")):
            publisher_counter[org] += 1
        if (org := extract_org_name(d, "sourceOrganization")):
            source_counter[org] += 1

        if (fac := extract_facility(d)):
            facility_counter[fac] += 1

        _, _, ongoing = parse_temporal(d)
        if ongoing:
            ongoing_total += 1
        for y in iter_temporal_years(d):
            year_counter[y] += 1

        if (ll := extract_lat_lon(d)):
            lats.append(ll[0])
            lons.append(ll[1])

    def top(counter: Counter, k: int = top_k) -> list[dict[str, Any]]:
        return [{"name": n, "count": c} for n, c in counter.most_common(k)]

    spatial = None
    if lats and lons:
        spatial = {
            "n_points": len(lats),
            "lat_min": min(lats), "lat_max": max(lats),
            "lon_min": min(lons), "lon_max": max(lons),
            "lat_mean": sum(lats) / len(lats),
            "lon_mean": sum(lons) / len(lons),
        }

    return {
        "total_datasets": total,
        "ongoing_datasets": ongoing_total,
        "top_variables": top(var_counter),
        "top_keywords": top(kw_counter),
        "top_providers": top(provider_counter),
        "top_publishers": top(publisher_counter),
        "top_source_organizations": top(source_counter),
        "facilities": top(facility_counter, 50),
        "datasets_per_year": dict(sorted(year_counter.items())),
        "spatial_summary": spatial,
        "variable_co_occurrence_top": [
            {"a": a, "b": b, "count": c}
            for (a, b), c in pair_counter.most_common(top_k)
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    stats = compute(Path(args.in_dir), top_k=args.top_k)
    save_json(stats, args.out)
    log.info("Stats salvo em %s (total=%d, ativos=%d)",
             args.out, stats["total_datasets"], stats["ongoing_datasets"])


if __name__ == "__main__":
    main()
