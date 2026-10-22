"""Leitura do catálogo de serviços ARM (data/arm_extra/catalog.json).

Usado pelo glossário (unidades e nomes longos oficiais) e pelo RAG (fichas
enriquecidas). O catálogo é gerado por `python -m src.collect.arm_extra`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.common.io import load_json

DEFAULT_CATALOG = "data/arm_extra/catalog.json"

# Preferência de nível ao escolher unidade/nome longo de uma variável:
# produtos de valor agregado e dados com QC antes dos brutos.
LEVEL_PREFERENCE = ["c1", "c2", "b1", "a1", "s1", "a0"]

_DIMENSIONLESS = {"unitless": "adimensional (unitless)",
                  "1": "1 (adimensional)"}
# outra unidade só é citada se aparecer em >= 20% dos produtos
ALT_UNIT_MIN_SHARE = 0.2


def load_catalog(path: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """"auto" = caminho padrão se existir; "none"/vazio = não usa."""
    if not path or str(path).lower() in ("none", "nenhum", "0"):
        return None, None
    p = DEFAULT_CATALOG if path == "auto" else path
    if Path(p).exists():
        return load_json(p), str(p)
    return None, None


def level_order(level: str) -> int:
    return LEVEL_PREFERENCE.index(level) if level in LEVEL_PREFERENCE else 99


def _clean(s: Any) -> str | None:
    s = str(s).strip() if s is not None else ""
    return s or None


def display_unit(u: Any) -> str | None:
    u = _clean(u)
    if u is None:
        return None
    return _DIMENSIONLESS.get(u.lower(), u)


def class_variables(c: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Variáveis de UMA classe (sem QC): nome em minúsculas ->
    {name, long_name, units, standard_name, levels}. Os atributos vêm do
    nível preferido (LEVEL_PREFERENCE); níveis seguintes só preenchem lacunas.
    """
    out: dict[str, dict[str, Any]] = {}
    levels = c.get("levels") or {}
    for lvl in sorted(levels, key=level_order):
        for v in levels[lvl].get("variables") or []:
            if v.get("is_qc") or not _clean(v.get("name")):
                continue
            k = v["name"].strip().lower()
            vals = {"long_name": _clean(v.get("long_name")),
                    "units": display_unit(v.get("units")),
                    "standard_name": _clean(v.get("standard_name"))}
            e = out.get(k)
            if e is None:
                out[k] = {"name": v["name"].strip(), **vals, "levels": [lvl]}
                continue
            e["levels"].append(lvl)
            for f, x in vals.items():
                if not e[f] and x:
                    e[f] = x
    return out


def variable_info(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """variável (minúsculas) -> {name, long_name, standard_name, units
    (Counter por produto), procs [classe.nível], classes (set)}.

    Cada PRODUTO (classe) conta uma vez: um produto com 4 níveis não pesa
    mais que um com 1 na escolha do nome longo e da unidade.
    """
    info: dict[str, dict[str, Any]] = {}
    for cls in sorted((catalog.get("classes") or {})):
        c = catalog["classes"][cls]
        for k, v in class_variables(c).items():
            e = info.setdefault(k, {"name": v["name"], "long_names": Counter(),
                                    "standard_names": Counter(),
                                    "units": Counter(), "procs": [],
                                    "classes": set()})
            if v["long_name"]:
                e["long_names"][v["long_name"]] += 1
            if v["standard_name"]:
                e["standard_names"][v["standard_name"]] += 1
            if v["units"]:
                e["units"][v["units"]] += 1
            e["procs"].extend(f"{cls}.{lvl}" for lvl in v["levels"])
            e["classes"].add(cls)
    for e in info.values():
        e["long_name"] = (e["long_names"].most_common(1)[0][0]
                          if e["long_names"] else None)
        e["standard_name"] = (e["standard_names"].most_common(1)[0][0]
                              if e["standard_names"] else None)
    return info


def unit_split(units: Counter) -> tuple[str | None, list[str]]:
    """(unidade mais frequente, outras com participação >= 20%)."""
    if not units:
        return None, []
    total = sum(units.values())
    (top, _), *rest = units.most_common()
    return top, [u for u, m in rest if m / total >= ALT_UNIT_MIN_SHARE]


def units_text(units: Counter) -> str | None:
    """"degC" ou "degC (também: K)"."""
    top, alts = unit_split(units)
    if top is None:
        return None
    return top + (f" (também: {', '.join(alts)})" if alts else "")
