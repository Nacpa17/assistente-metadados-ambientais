"""Métricas: a detecção de recusa precisa reconhecer TODAS as frases de recusa
que o próprio projeto usa (treino, gerador, benchmark e instrução do RAG)."""

import json
from pathlib import Path

import pytest

from assistente_metadados.common.io import iter_jsonl, load_yaml, save_json
from assistente_metadados.common.prompts import DEFAULT_RAG_INSTRUCTIONS
from assistente_metadados.eval.metrics import aggregate, contains_gold, is_refusal
from assistente_metadados.eval.rescore import rescore_file
from assistente_metadados.prompts.add_negatives import REFUSALS

ROOT = Path(__file__).resolve().parents[1]
GOLD = list(iter_jsonl(ROOT / "benchmarks" / "gold_qa.jsonl"))


def _project_refusals() -> list[str]:
    cfg = load_yaml(ROOT / "configs" / "dataset_v3.yaml")
    neg = cfg["negative_templates"]
    templates = [t.format(name="1twrmr") for t in neg["named"] + neg["contextual"]]
    gold_neg = [g["completion"] for g in GOLD
                if g["meta"]["kind"].startswith("negative")]
    # frase que a instrução do RAG pede para o modelo usar
    rag = "Essa informação não está disponível nos metadados."
    assert "não está disponível nos metadados" in DEFAULT_RAG_INSTRUCTIONS
    return list(REFUSALS) + templates + gold_neg + [rag]


@pytest.mark.parametrize("text", _project_refusals())
def test_toda_recusa_do_projeto_e_reconhecida(text):
    assert is_refusal(text), text


@pytest.mark.parametrize("text", [g["completion"] for g in GOLD
                                  if not g["meta"]["kind"].startswith("negative")])
def test_respostas_com_conteudo_nao_sao_recusa(text):
    assert not is_refusal(text)


def test_respostas_positivas_do_gerador_nao_sao_recusa():
    for s in ("Não. A coleta ocorreu de 1998-04-01 a 2010-12-31.",
              "Não há variáveis em comum entre os dois datasets.",
              "Nos metadados analisados, nenhum dataset de SGP consta como "
              "coleta ativa no momento."):
        assert not is_refusal(s), s


def test_contains_gold():
    assert contains_gold("O DOI é https://doi.org/10.5439/1025256.",
                         "https://doi.org/10.5439/1025256") == 1.0
    assert contains_gold("Os dados estão em NetCDF.", "NetCDF") == 1.0
    assert contains_gold("1998-04-02", "1998-04-01") == 0.0


def test_aggregate_metricas_novas():
    preds = ["O formato é NetCDF.", "Essa informação não está disponível.",
             "Não encontro essa informação nos metadados.", "NetCDF"]
    refs = ["NetCDF", "1998-04-01", "Não consta.", "NetCDF"]
    kinds = ["descriptive", "descriptive", "negative", "descriptive"]
    m = aggregate(preds, refs, kinds)
    assert m["refusal_accuracy"] == 1.0          # antes da correção: 0.0
    assert m["false_refusal_rate"] == round(1 / 3, 4)
    assert m["contains_gold"] == round(2 / 3, 4)
    assert m["exact_match"] == 0.25
    assert m["by_kind"]["negative"]["refusal_accuracy"] == 1.0
    assert m["by_kind"]["descriptive"]["n"] == 3
    assert set(m["token_f1_by_kind"]) == {"descriptive", "negative"}


def test_rescore_corrige_resultado_antigo(tmp_path):
    preds = [
        {"i": 1, "prompt": "Q", "gold": "Não consta.", "kind": "negative_x",
         "pred": "Esse detalhe não está descrito nos metadados.",
         "latency_s": 1.0},
        {"i": 2, "prompt": "Q", "gold": "NetCDF", "kind": "descriptive_fmt",
         "pred": "NetCDF", "latency_s": 2.0},
    ]
    old = {"refusal_accuracy": 0.0, "latency_mean_s": 1.5,
           "model_mode": "finetuned", "experiment_name": "exp"}
    p = tmp_path / "eval_exp_finetuned_1.json"
    save_json({"metrics": old, "predictions": preds}, p)
    new = rescore_file(p)
    assert new["refusal_accuracy"] == 1.0
    assert new["latency_mean_s"] == 1.5 and new["model_mode"] == "finetuned"
    out = json.loads((tmp_path / "eval_exp_finetuned_1_rescored.json")
                     .read_text(encoding="utf-8"))
    assert out["metrics_original"]["refusal_accuracy"] == 0.0
    assert json.loads(p.read_text(encoding="utf-8"))["metrics"] == old
