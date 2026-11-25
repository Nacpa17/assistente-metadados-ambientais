"""Auditoria do corpus coletado (data/raw_json): o que temos, o que falta e
o que se perdeu — rode antes de gerar o dataset ou montar o índice RAG.

Confere:
- arquivos e registros válidos (mesmo parser do gerador, parse_record);
- estado do coletor (.state.json): arquivos DIVIDIDOS por mais de uma URL
  (colisão de nomes do coletor antigo = registros sobrescritos), arquivos
  que o estado cita e não existem, arquivos que o estado não conhece;
- duplicatas por DOI, por nome e por conteúdo idêntico;
- cobertura dos campos (descrição, período, coordenadas, variáveis, ...);
- distribuição por sítio, facility, instrumento e produto;
- registros do Brasil (ex.: GoAmazon/MAO, Manacapuru-AM).

Uso (da raiz do projeto; não acessa a internet):
    python -m assistente_metadados.collect.audit
    python -m assistente_metadados.collect.audit --json-dir data/raw_json \
        --out data/analytics/auditoria_corpus.json
    python -m assistente_metadados.collect.audit --estrito    # código de saída 1 se houver perda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from assistente_metadados.common.io import list_json_files, load_json, save_json
from assistente_metadados.prompts.generator import Record, parse_record

PROVENANCE_KEY = "_fonte"          # gravado por assistente_metadados.collect.arm_collector
SITE_RE = re.compile(r"\(([A-Za-z]{3})\)")
FACILITY_RE = re.compile(r"\(([A-Z][0-9]{1,3})\)")
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.I)
BRAZIL_TEXT = re.compile(r"\b(brazil|brasil|manacapuru|manaus|amaz[oô]n\w*)\b",
                         re.I)
# MAO = ARM Mobile Facility da campanha GoAmazon (Manacapuru-AM). Não usamos
# caixa de coordenadas: ela pegaria campanhas vizinhas (ex.: CACTI, Argentina).
BRAZIL_SITES = {"mao"}

FIELDS: list[tuple[str, Callable[[Record], bool]]] = [
    ("descrição", lambda r: bool(r.description)),
    ("período", lambda r: bool(r.period_start)),
    ("coordenadas", lambda r: r.lat is not None and r.lon is not None),
    ("facility (about)", lambda r: bool(r.facility)),
    ("sítio/facility na citação", lambda r: bool(r.site_detail)),
    ("instrumento", lambda r: bool(r.instrument)),
    ("variáveis", lambda r: bool(r.variables)),
    ("DOI/URL de acesso", lambda r: bool(r.doi)),
    ("citação", lambda r: bool(r.citation)),
    ("criadores/mentores", lambda r: bool(r.creators)),
    ("palavras-chave", lambda r: bool(r.keywords)),
    ("formato", lambda r: bool(r.enc_format or r.data_format)),
    ("data de modificação", lambda r: bool(r.date_modified)),
]


def norm_doi(s: str | None) -> str | None:
    if not s:
        return None
    m = DOI_RE.search(s)
    return (m.group(0) if m else s).strip().rstrip(".").lower()


def sites_of(r: Record) -> list[str]:
    text = " ".join(filter(None, [r.facility, r.site_detail]))
    return sorted({m.lower() for m in SITE_RE.findall(text)})


def facilities_of(r: Record) -> list[str]:
    return sorted(set(FACILITY_RE.findall(r.site_detail or "")))


def is_brazil(r: Record) -> bool:
    if BRAZIL_SITES & set(sites_of(r)):
        return True
    text = " ".join(filter(None, [r.facility, r.site_detail, r.name]))
    return bool(BRAZIL_TEXT.search(text))


def content_hash(d: dict[str, Any]) -> str:
    core = {k: v for k, v in d.items() if k != PROVENANCE_KEY}
    return hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def audit_state(state_path: Path, json_dir: Path,
                file_names: set[str]) -> dict[str, Any]:
    if not state_path.exists():
        return {"present": False}
    urls = (load_json(state_path) or {}).get("urls") or {}
    by_file: dict[str, list[str]] = defaultdict(list)
    for url, info in urls.items():
        if (info or {}).get("filename"):
            by_file[info["filename"]].append(url)
    shared = {f: sorted(u) for f, u in sorted(by_file.items()) if len(u) > 1}
    return {
        "present": True,
        "path": str(state_path),
        "n_urls": len(urls),
        "n_files_referenced": len(by_file),
        "shared_files": shared,
        "n_overwritten_records": sum(len(u) - 1 for u in shared.values()),
        "missing_files": sorted(f for f in by_file
                                if not (json_dir / f).exists()),
        "files_not_in_state": sorted(file_names - set(by_file)),
    }


def audit(json_dir: str | Path, state: str | Path | None = None,
          top: int = 15) -> dict[str, Any]:
    json_dir = Path(json_dir)
    files = list_json_files(json_dir)
    invalid: list[dict[str, str]] = []
    recs: list[tuple[str, Record, dict]] = []
    n_prov = 0
    for p in files:
        try:
            d = load_json(p)
        except Exception as e:      # JSON quebrado
            invalid.append({"file": p.name, "reason": f"JSON inválido: {e}"})
            continue
        r = parse_record(d) if isinstance(d, dict) else None
        if r is None:
            invalid.append({"file": p.name, "reason": "sem campo name"})
            continue
        n_prov += isinstance(d.get(PROVENANCE_KEY), dict)
        recs.append((p.name, r, d))

    def dup_groups(key_fn) -> list[dict[str, Any]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for fname, r, d in recs:
            k = key_fn(r, d)
            if k:
                groups[k].append(fname)
        return [{"key": k, "files": sorted(v)}
                for k, v in sorted(groups.items()) if len(v) > 1]

    n = len(recs)
    coverage = {}
    for label, fn in FIELDS:
        k = sum(1 for _, r, _ in recs if fn(r))
        coverage[label] = {"n": k, "pct": round(100 * k / n, 1) if n else 0.0}
    n_vars = sum(len(r.variables) for _, r, _ in recs)
    n_vars_desc = sum(1 for _, r, _ in recs for v in r.variables
                      if v.get("description"))

    by_site: Counter = Counter()
    by_fac: Counter = Counter()
    by_instr: Counter = Counter()
    by_class: Counter = Counter()
    brazil = []
    for fname, r, _ in recs:
        for s in sites_of(r) or ["(sem sítio)"]:
            by_site[s.upper() if s != "(sem sítio)" else s] += 1
        for f in facilities_of(r):
            by_fac[f] += 1
        by_instr[r.instrument or "(sem instrumento)"] += 1
        by_class[r.short_code.lower()] += 1
        if is_brazil(r):
            brazil.append({"file": fname, "short_code": r.short_code,
                           "name": r.name, "facility": r.facility,
                           "site_detail": r.site_detail,
                           "period": r.period_display})

    state_path = Path(state) if state else json_dir / ".state.json"
    return {
        "json_dir": str(json_dir),
        "json_dir_abs": str(json_dir.resolve()),
        "n_files": len(files),
        "n_valid": n,
        "invalid": invalid,
        "n_with_provenance": n_prov,
        "state": audit_state(state_path, json_dir, {p.name for p in files}),
        "duplicates": {
            "doi": dup_groups(lambda r, d: norm_doi(r.doi)),
            "name": dup_groups(lambda r, d: r.name.strip().lower()),
            "content": dup_groups(lambda r, d: content_hash(d)),
        },
        "coverage": coverage,
        "variables": {"total": n_vars, "with_description": n_vars_desc,
                      "pct_with_description":
                          round(100 * n_vars_desc / n_vars, 1) if n_vars else 0.0},
        "by_site": dict(by_site.most_common()),
        "by_facility": dict(by_fac.most_common(top)),
        "by_instrument": dict(by_instr.most_common(top)),
        "n_product_classes": len(by_class),
        "classes_with_many_files": {c: k for c, k in by_class.most_common()
                                    if k > 1},
        "brazil": brazil,
    }


def recommendations(rep: dict[str, Any]) -> list[str]:
    if rep["n_files"] == 0:
        # sem corpus, as outras recomendações não fazem sentido
        return [f"Nenhum JSON em {rep['json_dir_abs']}. Confira se está na "
                "raiz do projeto (no Colab: célula 3) e se os JSON-LD estão "
                "em data/raw_json."]
    out = []
    st = rep["state"]
    if not st.get("present"):
        out.append("Sem .state.json: não dá para checar colisões de nome "
                   "(registros perdidos). Para conferir, colete numa pasta "
                   "nova e compare: python -m assistente_metadados.collect.arm_collector "
                   "--config configs/collect.yaml --out data/raw_json_v2 e "
                   "depois python -m assistente_metadados.collect.audit --json-dir "
                   "data/raw_json_v2.")
    elif st["n_overwritten_records"]:
        out.append(f"{st['n_overwritten_records']} registro(s) foram "
                   "sobrescritos por colisão de nomes. Rode: python -m "
                   "assistente_metadados.collect.arm_collector --config configs/collect.yaml "
                   "(baixa as páginas de novo, regrava só o necessário e "
                   "usa nomes únicos).")
    if st.get("missing_files"):
        out.append(f"{len(st['missing_files'])} arquivo(s) citados no estado "
                   "não existem; a próxima coleta os recria.")
    if rep["invalid"]:
        out.append(f"{len(rep['invalid'])} arquivo(s) inválidos ficam fora do "
                   "dataset — veja 'invalid' no relatório.")
    if rep["duplicates"]["content"]:
        out.append("Há arquivos com conteúdo idêntico: o mesmo registro "
                   "contaria duas vezes no treino/índice.")
    if not rep["brazil"]:
        out.append("Nenhum registro do Brasil no corpus: vale buscar os "
                   "produtos da campanha GoAmazon (sítio MAO) na ARM.")
    return out


def _bar(pct: float, width: int = 20) -> str:
    k = int(round(pct / 100 * width))
    return "#" * k + "." * (width - k)


def print_report(rep: dict[str, Any], top: int = 15) -> None:
    p = print
    p(f"\n== Auditoria do corpus: {rep['json_dir']} ==")
    if rep["n_files"] == 0:
        p(f"ATENÇÃO: nenhum JSON encontrado em {rep['json_dir_abs']} "
          f"(pasta atual: {Path.cwd()})")
    p(f"Arquivos: {rep['n_files']} | registros válidos: {rep['n_valid']} | "
      f"inválidos: {len(rep['invalid'])} | com procedência (_fonte): "
      f"{rep['n_with_provenance']}")
    st = rep["state"]
    if st.get("present"):
        p(f"Estado do coletor: {st['n_urls']} URLs | arquivos citados: "
          f"{st['n_files_referenced']} | divididos: {len(st['shared_files'])} "
          f"-> registros sobrescritos: {st['n_overwritten_records']} | "
          f"ausentes: {len(st['missing_files'])} | fora do estado: "
          f"{len(st['files_not_in_state'])}")
    else:
        p("Estado do coletor: .state.json não encontrado")
    dup = rep["duplicates"]
    p(f"Duplicatas: DOI {len(dup['doi'])} grupo(s) | nome {len(dup['name'])} | "
      f"conteúdo idêntico {len(dup['content'])}")
    p("\nCobertura dos campos:")
    for label, c in rep["coverage"].items():
        p(f"  {label:<27} {c['pct']:5.1f}%  {_bar(c['pct'])}  ({c['n']}/"
          f"{rep['n_valid']})")
    v = rep["variables"]
    p(f"  {'variáveis com descrição':<27} {v['pct_with_description']:5.1f}%  "
      f"{_bar(v['pct_with_description'])}  ({v['with_description']}/{v['total']})")
    p(f"\nProdutos (classes) distintos: {rep['n_product_classes']}")
    p("Sítios: " + " | ".join(f"{s} {k}" for s, k in
                              list(rep["by_site"].items())[:top]))
    p("Facilities: " + " | ".join(f"{s} {k}" for s, k in
                                  rep["by_facility"].items()))
    p(f"\nBrasil: {len(rep['brazil'])} registro(s)")
    for b in rep["brazil"][:top]:
        p(f"  - {b['short_code']}: {b['name'][:80]}")
    recs = recommendations(rep)
    if recs:
        p("\nRecomendações:")
        for r in recs:
            p(f"  * {r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default="data/raw_json")
    ap.add_argument("--state", default=None,
                    help="estado do coletor (padrão: <json-dir>/.state.json)")
    ap.add_argument("--out", default="data/analytics/auditoria_corpus.json")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--estrito", action="store_true",
                    help="sai com código 1 se houver registros perdidos, "
                         "inválidos ou duplicados por conteúdo")
    a = ap.parse_args()

    rep = audit(a.json_dir, a.state, top=a.top)
    rep["recommendations"] = recommendations(rep)
    save_json(rep, a.out)
    print_report(rep, top=a.top)
    print(f"\nRelatório completo: {a.out}")
    if a.estrito and (rep["n_files"] == 0
                      or rep["state"].get("n_overwritten_records")
                      or rep["invalid"] or rep["duplicates"]["content"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
