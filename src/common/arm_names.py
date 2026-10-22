"""Convenção de nomes de datastream da ARM (sem dependências externas).

    <sítio (3 letras)><classe do produto><facility>.<nível de dados>

    sgpmetE13.b1      -> sítio sgp | classe met      | facility E13 | nível b1
    sgp30baebbrE39.c1 -> sítio sgp | classe 30baebbr | facility E39 | nível c1
    maoarmbeatmM1.c1  -> sítio mao | classe armbeatm | facility M1  | nível c1

A classe é minúscula e a facility começa com letra maiúscula, então a divisão
não é ambígua. O "proc" (classe.nível, ex.: "kazrge.a1") identifica o DOD.
"""

from __future__ import annotations

import re

DATASTREAM_RE = re.compile(
    r"^(?P<site>[a-z]{3})(?P<cls>[a-z0-9]+?)(?P<facility>[A-Z][0-9]{1,3})"
    r"\.(?P<level>[a-z0-9]{2})$")


def parse_datastream(name: str) -> dict[str, str] | None:
    """Decompõe um nome de datastream ARM nas suas partes (ou None)."""
    m = DATASTREAM_RE.match((name or "").strip())
    if not m:
        return None
    d = m.groupdict()
    return {"datastream": name.strip(), "site": d["site"], "class": d["cls"],
            "facility": d["facility"], "level": d["level"],
            "proc": f"{d['cls']}.{d['level']}"}
