"""Avalia a RECUPERAÇÃO do RAG (sem modelo, sem GPU): a ficha certa aparece?

Gabarito: o próprio dataset de treino. Cada par gerado por
assistente_metadados/prompts/generator.py traz em meta.dataset o nome do dataset de origem,
então sabemos qual ficha a busca deveria devolver. Medimos:

- Recall@k : fração das perguntas cuja ficha-alvo está entre as k primeiras
             (multi-hop: as DUAS fichas precisam estar entre as max(k, 2)
             primeiras — com dois alvos, "top-1" seria impossível)
- MRR@10   : média de 1/posição da ficha-alvo (1.0 = sempre em 1º lugar)

por tipo de pergunta (descriptive_doi, descriptive_period, ...). Perguntas
"contextuais" (sem o nome do dataset, ex.: "Qual o DOI deste dataset?") são
ignoradas: sem identificar o dataset não há o que recuperar.

Também gera perguntas de glossário a partir dos MESMOS templates do gerador
(configs/dataset_v3.yaml) e mede quanto as perguntas fora de escopo
(add_negatives) "casam" com algo — útil para calibrar --min-score.

Uso:
    python -m assistente_metadados.rag.eval_retrieval
    python -m assistente_metadados.rag.eval_retrieval --per-kind 0     # todas as perguntas
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from assistente_metadados.common.io import iter_jsonl, load_yaml, save_json, setup_logger
from assistente_metadados.rag.index import Index
from assistente_metadados.rag.text import phrase

log = setup_logger("rag.eval_retrieval")

DEFAULT_KS = (1, 3, 5, 10)


def _percentiles(xs: list[float], ps=(10, 50, 90)) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{p}": round(s[min(len(s) - 1, int(p / 100 * len(s)))], 3)
            for p in ps}


def build_queries(index: Index, dataset_path: str | Path | None,
                  gloss_templates: list[str], acronym_templates: list[str],
                  per_kind: int, seed: int) -> tuple[dict[str, list[dict]], dict]:
    """Agrupa as consultas por tipo: [{query, targets:[doc_id,...]}]."""
    rng = random.Random(seed)
    name_to_id: dict[str, str] = {}
    code_of: dict[str, str] = {}
    for d in index.docs:
        if d.kind == "dataset":
            name_to_id[d.meta["dataset"]] = d.id
            code_of[d.meta["dataset"]] = d.meta.get("short_code") or ""

    by_kind: dict[str, list[dict]] = defaultdict(list)
    stats = {"pairs": 0, "skipped_contextual": 0, "skipped_unknown_dataset": 0,
             "skipped_other": 0, "out_of_scope": 0}
    oos: list[str] = []

    if dataset_path and Path(dataset_path).exists():
        for ex in iter_jsonl(dataset_path):
            stats["pairs"] += 1
            meta = ex.get("meta") or {}
            kind = meta.get("kind", "unknown")
            ds = meta.get("dataset") or ""
            q = ex.get("prompt") or ""
            if kind == "negative_out_of_scope":
                stats["out_of_scope"] += 1
                oos.append(q)
                continue
            if not ds or ds.startswith("_"):
                stats["skipped_other"] += 1      # corpus/glossário/negativos
                continue
            names = ds.split("|")
            ids = [name_to_id.get(n) for n in names]
            if not all(ids):
                stats["skipped_unknown_dataset"] += 1
                continue
            qp = phrase(q)
            named = all(
                phrase(n) in qp or (code_of.get(n) and phrase(code_of[n]) in qp)
                for n in names)
            if not named:
                stats["skipped_contextual"] += 1
                continue
            by_kind[kind].append({"query": q, "targets": ids})
    elif dataset_path:
        log.warning("Dataset %s não encontrado; avaliando só o glossário.",
                    dataset_path)

    # glossário: mesmas formulações do gerador
    for d in index.docs:
        if d.kind != "glossary":
            continue
        term = d.meta.get("term") or d.title
        if gloss_templates:
            by_kind["glossary_definition*"].append({
                "query": rng.choice(gloss_templates).format(term=term),
                "targets": [d.id]})
        if acronym_templates and d.meta.get("expansao"):
            by_kind["glossary_acronym*"].append({
                "query": rng.choice(acronym_templates).format(term=term),
                "targets": [d.id]})

    if per_kind > 0:
        for k, qs in by_kind.items():
            if len(qs) > per_kind:
                rng.shuffle(qs)
                by_kind[k] = qs[:per_kind]
    stats["out_of_scope_queries"] = oos
    return by_kind, stats


def evaluate(index: Index, by_kind: dict[str, list[dict]], ks=DEFAULT_KS,
             expand: bool = True, focus: bool = True) -> dict[str, Any]:
    max_k = max(ks)
    per_kind: dict[str, dict[str, Any]] = {}
    totals = {"n": 0, "rr": 0.0, **{f"r@{k}": 0 for k in ks}}
    top1_scores: list[float] = []
    failures: list[dict] = []

    for kind in sorted(by_kind):
        acc = {"n": 0, "rr": 0.0, **{f"r@{k}": 0 for k in ks}}
        for item in by_kind[kind]:
            hits = index.search(item["query"], k=max_k, expand=expand,
                                focus=focus)
            ids = [h.doc.id for h in hits]
            ranks = [ids.index(t) + 1 if t in ids else None
                     for t in item["targets"]]
            worst = None if None in ranks else max(ranks)
            best = min((r for r in ranks if r), default=None)
            acc["n"] += 1
            acc["rr"] += (1.0 / best) if best else 0.0
            n_t = len(item["targets"])
            for k in ks:
                # com 2 alvos (multi-hop) o "top-1" é impossível: usamos
                # top-max(k, nº de alvos)
                acc[f"r@{k}"] += int(worst is not None and worst <= max(k, n_t))
            if hits:
                top1_scores.append(hits[0].score)
            # falha = algum alvo fora das primeiras n_alvos posições
            if (worst is None or worst > n_t) and len(failures) < 200:
                failures.append({"kind": kind, "query": item["query"],
                                 "targets": item["targets"],
                                 "got": ids[:3], "ranks": ranks})
        per_kind[kind] = {
            "n": acc["n"],
            "mrr@10": round(acc["rr"] / max(1, acc["n"]), 4),
            **{f"recall@{k}": round(acc[f"r@{k}"] / max(1, acc["n"]), 4)
               for k in ks}}
        for key in totals:
            totals[key] += acc[key]

    n = max(1, totals["n"])
    overall = {"n": totals["n"], "mrr@10": round(totals["rr"] / n, 4),
               **{f"recall@{k}": round(totals[f"r@{k}"] / n, 4) for k in ks}}
    return {"overall": overall, "by_kind": per_kind,
            "positive_top1_bm25": _percentiles(top1_scores),
            "failures_sample": failures}


def out_of_scope_report(index: Index, queries: list[str],
                        expand: bool = True) -> dict[str, Any]:
    """Quanto as perguntas fora de escopo 'puxam' documentos (ideal: pouco)."""
    uniq = sorted(set(queries))
    if not uniq:
        return {}
    exact = 0
    any_hit = 0
    top1: list[float] = []
    for q in uniq:
        hits = index.search(q, k=1, expand=expand)
        if hits:
            any_hit += 1
            top1.append(hits[0].score)
            exact += int(bool(hits[0].boost))
    return {"n_unique": len(uniq),
            "with_any_hit": round(any_hit / len(uniq), 4),
            "with_exact_match": round(exact / len(uniq), 4),
            "top1_bm25": _percentiles(top1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/rag/index.json")
    ap.add_argument("--dataset", default="data/processed/dataset_v3.jsonl")
    ap.add_argument("--generator-config", default="configs/dataset_v3.yaml",
                    help="de onde vêm os templates de pergunta do glossário")
    ap.add_argument("--per-kind", type=int, default=300,
                    help="máx. de perguntas por tipo (0 = todas)")
    ap.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-expand", action="store_true",
                    help="desliga a expansão PT->EN (para comparação)")
    ap.add_argument("--no-focus", action="store_true",
                    help="desliga o filtro de foco (para comparação)")
    ap.add_argument("--out-dir", default="benchmarks/results")
    a = ap.parse_args()

    index = Index.load(a.index)
    gcfg = load_yaml(a.generator_config) if Path(a.generator_config).exists() else {}
    gp = (gcfg or {}).get("glossary_prompts", {})
    by_kind, stats = build_queries(index, a.dataset, gp.get("definition", []),
                                   gp.get("acronym", []), a.per_kind, a.seed)
    oos = stats.pop("out_of_scope_queries")
    log.info("Perguntas avaliadas: %d em %d tipos | ignoradas: contextuais=%d, "
             "corpus/glossário/negativos=%d, dataset fora do índice=%d",
             sum(len(v) for v in by_kind.values()), len(by_kind),
             stats["skipped_contextual"], stats["skipped_other"],
             stats["skipped_unknown_dataset"])

    t0 = time.time()
    res = evaluate(index, by_kind, ks=sorted(set(a.k)),
                   expand=not a.no_expand, focus=not a.no_focus)
    res["out_of_scope"] = out_of_scope_report(index, oos, expand=not a.no_expand)
    res["stats"] = stats
    res["settings"] = {"index": a.index, "dataset": a.dataset,
                       "per_kind": a.per_kind, "seed": a.seed,
                       "expand_pt_en": not a.no_expand,
                       "focus": not a.no_focus,
                       "seconds": round(time.time() - t0, 1)}

    ks = sorted(set(a.k))
    head = f"{'tipo de pergunta':<34}{'n':>6}" + "".join(
        f"{'R@' + str(k):>8}" for k in ks) + f"{'MRR':>8}"
    print("\n" + head + "\n" + "-" * len(head))
    for kind, m in res["by_kind"].items():
        print(f"{kind:<34}{m['n']:>6}" + "".join(
            f"{m[f'recall@{k}']:>8.3f}" for k in ks) + f"{m['mrr@10']:>8.3f}")
    o = res["overall"]
    print("-" * len(head))
    print(f"{'GERAL':<34}{o['n']:>6}" + "".join(
        f"{o[f'recall@{k}']:>8.3f}" for k in ks) + f"{o['mrr@10']:>8.3f}")
    if res["out_of_scope"]:
        r = res["out_of_scope"]
        print(f"\nFora de escopo ({r['n_unique']} perguntas únicas): "
              f"{r['with_any_hit']:.0%} puxam algum documento, "
              f"{r['with_exact_match']:.0%} com identificador exato | "
              f"BM25 do 1º: {r['top1_bm25']}")
        print(f"Positivas — BM25 do 1º: {res['positive_top1_bm25']} "
              f"(use para escolher rag.min_score)")
    print("(* = perguntas geradas a partir do glossário com os templates do gerador)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir) / f"retrieval_{ts}.json"
    save_json(res, out)
    log.info("Relatório salvo em %s (inclui amostra de erros em "
             "'failures_sample')", out)


if __name__ == "__main__":
    main()
