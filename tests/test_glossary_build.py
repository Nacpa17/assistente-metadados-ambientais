"""Glossário com unidades e nomes longos oficiais dos DODs (--extra)."""

import json
import random

import pytest
import yaml

from conftest import ROOT
from assistente_metadados.common.io import save_json
from assistente_metadados.glossary.build import (build_glossary, harvest_dod, load_curated,
                                useful_long_name)
from assistente_metadados.prompts.generator import GlossaryQA, QABuilder

LEGACY_KEYS = {"term", "aliases", "category", "unit", "definition", "uso",
               "expansao", "source", "n_datasets"}


@pytest.fixture()
def mini_curated(tmp_path):
    """Glossário curado mínimo: só 'reflectivity' (o resto vira auto)."""
    p = tmp_path / "curado.yaml"
    p.write_text(yaml.safe_dump({"entries": [{
        "term": "reflectivity", "aliases": [], "category": "radar",
        "unit": None, "definition": "Fator de refletividade do radar.",
        "uso": None, "expansao": None}]}, allow_unicode=True),
        encoding="utf-8")
    return p


def by_term(out):
    return {e["term"]: e for e in out["entries"]}


def test_sem_extra_mantem_formato_da_fase2(corpus_dir, mini_curated):
    out = build_glossary(corpus_dir, mini_curated, min_count=2, extra=None)
    assert out["n_dod"] == 0 and "extra" not in out
    auto = [e for e in out["entries"] if e["source"] == "auto"]
    assert auto, "o mini-corpus deveria gerar entradas automáticas"
    for e in auto:
        assert set(e) == LEGACY_KEYS
        assert e["unit"] is None
    assert not [e for e in out["entries"] if e["source"] == "dod"]


def test_com_extra_unidade_oficial_e_variaveis_so_do_dod(corpus_dir, mini_curated,
                                                       catalog_path):
    out = build_glossary(corpus_dir, mini_curated, min_count=2,
                         extra=str(catalog_path))
    t = by_term(out)
    # auto (descrição do JSON-LD) + unidade do DOD
    assert t["temp"]["source"] == "auto"
    assert t["temp"]["definition"] == "Air temperature"
    assert (t["temp"]["unit"], t["temp"]["unit_source"]) == ("degC", "dod")
    assert t["mixing_ratio"]["unit"] == "g/kg"
    # sem DOD no servidor falso: segue como antes
    assert t["pres"]["unit"] is None and "unit_source" not in t["pres"]
    # só no DOD (não está em nenhum JSON-LD do corpus)
    assert t["station_id"]["source"] == "dod"
    assert t["station_id"]["definition"] == "Station identifier"
    assert t["station_id"]["unit"] is None           # variável de texto
    # QC nunca entra; curado nunca é alterado
    assert "qc_rh" not in t
    assert t["reflectivity"] == {**load_curated(mini_curated)[0]}
    assert out["n_dod"] == 1
    assert out["extra"]["n_units_from_dod"] >= 3
    json.dumps(out)


def test_definicao_do_dod_quando_o_json_ld_nao_tem(tmp_path, mini_curated):
    corpus = tmp_path / "raw_json"
    corpus.mkdir()
    for i in range(2):
        save_json({"name": f"Teste {i}", "variableMeasured": [
            {"name": "temp"}, {"name": "time"}]}, corpus / f"t{i}.json")
    cat = {"classes": {"x": {"levels": {"b1": {"variables": [
        {"name": "temp", "long_name": "Air temperature", "units": "degC"},
        {"name": "time", "long_name": "Time", "units": "s"}]}}}}}
    save_json(cat, tmp_path / "cat.json")
    out = build_glossary(corpus, mini_curated, min_count=2,
                         extra=str(tmp_path / "cat.json"))
    t = by_term(out)
    assert t["temp"]["source"] == "dod"
    assert t["temp"]["definition"] == "Air temperature"
    assert "time" not in t       # nome longo só repete o nome: descartado


def test_nome_longo_util():
    assert useful_long_name("base_time", "Base time in Epoch")
    assert not useful_long_name("time", "time")
    assert not useful_long_name("lat", " LAT ")
    assert not useful_long_name("x", None)


def test_limites_das_entradas_do_dod():
    from collections import Counter
    dod = {f"v{i}": {"name": f"v{i}", "long_name": f"Variable number {i}",
                     "units": Counter({"m": 1}) if i % 2 else Counter(),
                     "standard_name": None, "classes": set(range(i % 3 + 1)),
                     "procs": []} for i in range(10)}
    sel = harvest_dod(dod, covered={"v8"}, min_products=2, max_dod=3)
    assert len(sel) == 3
    assert all(e["n_products_dod"] >= 2 for e in sel)
    assert "v8" not in {e["term"] for e in sel}
    # mais produtos primeiro; com unidade antes de sem unidade
    assert [e["n_products_dod"] for e in sel] == sorted(
        [e["n_products_dod"] for e in sel], reverse=True)


def _qa(entries):
    cfg = yaml.safe_load((ROOT / "configs" / "dataset_v3.yaml")
                         .read_text(encoding="utf-8"))
    rng = random.Random(0)
    return GlossaryQA(entries, cfg, rng, QABuilder(cfg, rng), by_var={}).build()


def test_respostas_do_gerador_citam_o_dod():
    pairs = _qa([
        {"term": "station_id", "aliases": [], "definition": "Station identifier",
         "source": "dod", "unit": "adimensional (unitless)",
         "unit_source": "dod", "unit_alternatives": []},
        {"term": "temp", "aliases": [], "definition": "Air temperature",
         "source": "auto", "unit": "degC", "unit_source": "dod",
         "unit_alternatives": ["K"]},
        {"term": "pwv", "aliases": [], "definition": "Vapor d'água precipitável.",
         "source": "curated", "unit": "cm"},
    ])
    defs = [p["completion"] for p in pairs
            if p["meta"]["kind"] == "glossary_definition"]
    units = [p["completion"] for p in pairs
             if p["meta"]["kind"] == "glossary_unit"]
    assert ('Nos metadados oficiais da ARM (DOD), "station_id" tem o nome '
            'longo "Station identifier".') in defs
    assert ("Segundo os DODs oficiais da ARM, station_id é adimensional "
            "(sem unidade física).") in units
    assert ("Segundo os DODs oficiais da ARM, temp é expresso em degC; "
            "alguns produtos usam K.") in units
    assert "pwv é tipicamente expresso em cm." in units     # curado: igual
