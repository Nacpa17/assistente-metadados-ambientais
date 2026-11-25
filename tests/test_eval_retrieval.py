"""Integração: gerador REAL do projeto -> perguntas -> avaliação da busca."""

from pathlib import Path

import pytest

from assistente_metadados.common.io import load_yaml
from assistente_metadados.prompts.generator import run as run_generator
from assistente_metadados.rag.eval_retrieval import build_queries, evaluate, out_of_scope_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def generated_dataset(corpus_dir, tmp_path_factory):
    import yaml
    d = tmp_path_factory.mktemp("gen")
    cfg = load_yaml(ROOT / "configs" / "dataset_v3.yaml")
    cfg.update({"input_json_dir": str(corpus_dir),
                "output_path": str(d / "dataset_v3.jsonl"),
                "glossary_path": str(d / "nao_existe.json")})
    p = d / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return run_generator(str(p))


def test_recall_no_dataset_gerado(index, generated_dataset):
    gcfg = load_yaml(ROOT / "configs" / "dataset_v3.yaml")["glossary_prompts"]
    by_kind, stats = build_queries(index, generated_dataset,
                                   gcfg["definition"], gcfg["acronym"],
                                   per_kind=0, seed=1)
    assert stats["skipped_contextual"] > 0          # "deste dataset" ignoradas
    assert "descriptive_doi" in by_kind and "glossary_definition*" in by_kind
    res = evaluate(index, by_kind)
    assert res["overall"]["n"] > 100
    assert res["overall"]["recall@3"] >= 0.98
    for kind in ("descriptive_doi", "descriptive_period", "descriptive_location"):
        assert res["by_kind"][kind]["recall@1"] == 1.0, kind
    # multi-hop: dois alvos contam como acerto no top-2
    mh = [k for k in res["by_kind"] if k.startswith("multi_hop")]
    assert mh and all(res["by_kind"][k]["recall@1"] == 1.0 for k in mh)


def test_relatorio_fora_de_escopo(index):
    r = out_of_scope_report(index, ["Qual a cotação do dólar hoje?",
                                    "Qual a altura da torre Eiffel?"])
    assert r["n_unique"] == 2
    assert r["with_exact_match"] == 0.0
