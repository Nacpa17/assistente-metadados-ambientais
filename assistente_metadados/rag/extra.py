"""Acrescenta ao índice RAG o catálogo de serviços oficiais da ARM.

O catálogo vem de `python -m assistente_metadados.collect.arm_extra` (DOD, DQR, citações e
sítios). Aqui ele vira conhecimento consultável:

- fichas de DATASET ganham: níveis de processamento (DOD), datastreams
  CONFIRMADOS (com DOI), resumo dos relatórios de qualidade e, nas variáveis,
  NOME LONGO e UNIDADE oficiais;
- fichas de VARIÁVEL ganham nome longo e unidade; variáveis que só aparecem
  nos DODs ganham ficha própria;
- cada DQR vira uma ficha ("dqr"): categoria, período, variáveis afetadas,
  descrição e sugestão — informação que MUDA com o tempo, por isso fica no
  RAG e não no treino;
- cada sítio vira uma ficha ("site") com as facilities e coordenadas.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from assistente_metadados.common.arm_catalog import (class_variables, level_order, units_text,
                                    variable_info)
from assistente_metadados.common.arm_names import parse_datastream
from assistente_metadados.prompts.generator import cap_join
from assistente_metadados.rag.documents import Doc

# Significado das categorias segundo a página de DQRs da ARM
# (arm.gov/data/data-quality-program/data-quality-reporting).
DQR_CATEGORY_PT = {
    "incorrect": "Incorrect — dados incorretos: não devem ser usados",
    "suspect": "Suspect — dados suspeitos: exigem triagem adicional antes do uso",
    "missing": "Missing — período com dados ausentes",
    "transparent": "Transparent — evento informativo, sem impacto na qualidade",
}


def _class_of(doc: Doc) -> str:
    code = str(doc.meta.get("short_code") or "").lower()
    return code.split(".", 1)[0]


def merge_dqrs(dqrs: list[dict]) -> list[dict]:
    """Um mesmo DQR pode valer para vários datastreams (ex.: duas facilities):
    junta numa entrada só, com a lista de datastreams e de classes."""
    merged: dict[str, dict] = {}
    for d in dqrs:
        m = merged.get(d["id"])
        if m is None:
            m = dict(d)
            m["datastreams"] = []
            m["classes"] = []
            merged[d["id"]] = m
        if d["datastream"] not in m["datastreams"]:
            m["datastreams"].append(d["datastream"])
        if d.get("class") and d["class"] not in m["classes"]:
            m["classes"].append(d["class"])
    return [merged[k] for k in sorted(merged)]


def _dqr_summary(dqrs: list[dict]) -> str | None:
    if not dqrs:
        return None
    cats = Counter(d["category"] for d in dqrs)
    recent = sorted(dqrs, key=lambda d: (d["dates"][0]["start"] or "")
                    if d.get("dates") else "", reverse=True)[:3]
    ex = "; ".join(f"{d['id']} ({d['category']}: {d.get('subject') or 's/ assunto'})"
                   for d in recent)
    return (f"Relatórios de qualidade (DQR): {len(dqrs)} ("
            + ", ".join(f"{n} {c}" for c, n in cats.most_common())
            + f"). Mais recentes: {ex}.")


def _enrich_dataset(doc: Doc, c: dict[str, Any], dqrs: list[dict]) -> None:
    lines: list[str] = []
    levels = c.get("levels") or {}
    if levels:
        lines.append("Níveis de processamento (DOD oficial): " + ", ".join(
            f"{lvl} (versão {levels[lvl].get('version')}, "
            f"{len(levels[lvl].get('variables') or [])} variáveis)"
            for lvl in sorted(levels, key=level_order)))
    confirmed = [d for d in c.get("datastreams") or [] if d.get("confirmed")]
    if confirmed:
        lines.append("Datastreams confirmados: " + cap_join([
            d["datastream"] + (f" (DOI {d['doi']})" if d.get("doi") else "")
            for d in confirmed], 8, "; "))
    if (s := _dqr_summary(dqrs)):
        lines.append(s)
    n_core = int(doc.meta.get("n_core", len(doc.lines)))
    doc.lines[n_core:n_core] = lines
    doc.meta["n_core"] = n_core + len(lines)

    # variáveis: unidade/nome longo oficiais + variáveis que só o DOD lista
    cv = class_variables(c)
    variables = [dict(v) for v in doc.meta.get("variables") or []]
    known = set()
    for v in variables:
        k = v["name"].lower()
        known.add(k)
        if k in cv:
            v.setdefault("units", cv[k]["units"])
            if not v.get("description") and cv[k]["long_name"]:
                v["description"] = cv[k]["long_name"]
    for k, v in cv.items():
        if k not in known:
            variables.append({"name": v["name"],
                              "description": v["long_name"] or "",
                              "units": v["units"], "source": "dod"})
    doc.meta["variables"] = variables
    doc.search_text += " " + " ".join(
        f"{v['name']} {v['long_name'] or ''}" for v in cv.values())
    for d in confirmed:
        doc.keys.append(d["datastream"])
        if d.get("doi"):
            doc.keys.append(d["doi"])


def _where(ds: str) -> str:
    p = parse_datastream(ds)
    if not p:
        return ds
    return (f"{ds} (produto {p['class']}, sítio {p['site'].upper()}, "
            f"facility {p['facility']}, nível {p['level']})")


def dqr_doc(d: dict[str, Any]) -> Doc:
    """Ficha de um DQR (já fundido: d["datastreams"] pode ter vários)."""
    period = "; ".join(f"{x.get('start')} a {x.get('end')}"
                       for x in d.get("dates") or [])
    dss = d.get("datastreams") or [d["datastream"]]
    L = [f"Relatório de qualidade (DQR): {d['id']}",
         ("Datastream: " if len(dss) == 1 else "Datastreams: ")
         + "; ".join(_where(x) for x in dss[:6])
         + (f"; e mais {len(dss) - 6}" if len(dss) > 6 else ""),
         "Categoria: " + DQR_CATEGORY_PT.get(d["category"], d["category"])]
    if d.get("subject"):
        L.append(f"Assunto: {d['subject']}")
    if period:
        L.append(f"Período afetado: {period}")
    if d.get("variables"):
        L.append(f"Variáveis afetadas: {cap_join(list(d['variables']), 15)}")
    if d.get("description"):
        L.append(f"Descrição: {d['description']}")
    if d.get("suggestions"):
        L.append(f"Sugestão: {d['suggestions']}")
    L.append(f"Página do DQR: {d.get('url')}")
    start = (d.get("dates") or [{}])[0].get("start") or ""
    classes = d.get("classes") or ([d["class"]] if d.get("class") else [])
    return Doc(
        id=f"dqr:{d['id']}", kind="dqr", title=f"DQR {d['id']} — {dss[0]}",
        lines=L,
        search_text=" ".join(filter(None, [
            d.get("subject"), d.get("description"), d.get("suggestions"),
            " ".join(d.get("variables") or []), " ".join(dss),
            " ".join(classes), d["category"], "quality report dqr"])),
        keys=[d["id"]] + dss,
        meta={"dqr_id": d["id"], "datastreams": dss, "classes": classes,
              "category": d["category"], "start": start})


def site_docs(sites: list[dict[str, Any]]) -> list[Doc]:
    by_site: dict[str, list[dict]] = defaultdict(list)
    for s in sites:
        if s.get("site_code"):
            by_site[str(s["site_code"]).lower()].append(s)
    docs = []
    for code, facs in sorted(by_site.items()):
        types = sorted({f["site_type"] for f in facs if f.get("site_type")})
        parts = []
        for f in sorted(facs, key=lambda f: str(f.get("facility_code"))):
            if f.get("latitude") is not None:
                parts.append(f"{f.get('facility_code')} (latitude "
                             f"{f['latitude']}, longitude {f['longitude']})")
            else:
                parts.append(f"{f.get('facility_code')} (coordenadas não "
                             f"informadas)")
        L = [f"Sítio ARM: {code.upper()}"]
        if types:
            L.append(f"Tipo: {', '.join(types)}")
        L.append(f"Facilities ({len(facs)}): {cap_join(parts, 20, '; ')}")
        docs.append(Doc(
            id=f"site:{code}", kind="site", title=f"Sítio ARM {code.upper()}",
            lines=L,
            search_text=" ".join([code, "site sitio facilities coordenadas"]
                                 + [str(f.get("facility_code")) for f in facs]),
            keys=[code.upper()],
            meta={"site_code": code, "n_facilities": len(facs)}))
    return docs


def apply_extra(docs: list[Doc], extra: dict[str, Any] | None) -> list[Doc]:
    """Enriquece as fichas existentes e acrescenta as novas (dqr, site)."""
    if not extra:
        return docs
    classes = extra.get("classes") or {}
    dqrs = merge_dqrs(extra.get("dqrs") or [])
    dqrs_by_class: dict[str, list[dict]] = defaultdict(list)
    for d in dqrs:
        for c in d["classes"]:
            dqrs_by_class[c].append(d)

    for doc in docs:
        if doc.kind == "dataset" and (cls := _class_of(doc)) in classes:
            _enrich_dataset(doc, classes[cls], dqrs_by_class.get(cls, []))

    vinfo = variable_info(extra)
    var_docs = {d.meta.get("variable", "").lower(): d for d in docs
                if d.kind == "variable"}
    datasets_by_class: dict[str, list[Doc]] = defaultdict(list)
    for d in docs:
        if d.kind == "dataset":
            datasets_by_class[_class_of(d)].append(d)
    new_docs: list[Doc] = []
    for k, e in sorted(vinfo.items()):
        extra_lines = []
        if e["long_name"]:
            extra_lines.append(f"Nome longo (DOD oficial): {e['long_name']}")
        if (u := units_text(e["units"])):
            extra_lines.append(f"Unidade (DOD oficial): {u}")
        if e.get("standard_name"):
            extra_lines.append(f"Nome padrão CF (standard_name): "
                               f"{e['standard_name']}")
        extra_lines.append("Produtos que a definem (DOD): "
                           + cap_join(sorted(set(e["procs"])), 10))
        extra_search = " ".join(filter(None, [e["long_name"],
                                              e.get("standard_name")]))
        if k in var_docs:
            var_docs[k].lines.extend(extra_lines)
            var_docs[k].search_text += f" {extra_search}"
            if e.get("standard_name"):
                var_docs[k].keys.append(e["standard_name"])
            continue
        dsets = [d for c in sorted(e["classes"]) for d in datasets_by_class.get(c, [])]
        L = [f"Variável: {e['name']}"] + extra_lines
        if dsets:
            L.append("Datasets do corpus com esse produto: " + "; ".join(
                d.title + (f" (acesso: {d.meta['doi']})" if d.meta.get("doi")
                           else "") for d in dsets[:5]) + ".")
        new_docs.append(Doc(
            id=f"var:{k}", kind="variable", title=f"Variável {e['name']}",
            lines=L, search_text=f"{e['name']} {extra_search}",
            keys=[e["name"]] + ([e["standard_name"]]
                                if e.get("standard_name") else []),
            meta={"variable": e["name"], "n_datasets": len(dsets),
                  "source": "dod"}))

    new_docs.extend(dqr_doc(d) for d in dqrs)
    new_docs.extend(site_docs(extra.get("sites") or []))
    return docs + new_docs
