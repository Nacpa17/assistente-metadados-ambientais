"""Catálogo de serviços ARM (DOD/DQR/citação/sítios) dentro do RAG.

O catálogo é montado com o servidor FALSO de tests/fake_arm.py (formatos
reais do ACT, dados inventados) a partir do mini-corpus sintético.
"""

from collections import Counter

import pytest

from conftest import CURATED_GLOSSARY, EXPANSIONS
from assistente_metadados.common.arm_catalog import (class_variables, display_unit, unit_split,
                                    units_text, variable_info)
from assistente_metadados.common.io import save_json
from assistente_metadados.rag.build_index import build, load_extra
from assistente_metadados.rag.context import build_context
from assistente_metadados.rag.extra import apply_extra, merge_dqrs
from assistente_metadados.rag.index import Index


@pytest.fixture(scope="session")
def index_extra(corpus_dir, catalog_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("rag_extra") / "index.json"
    build(str(corpus_dir), str(CURATED_GLOSSARY), str(EXPANSIONS), str(out),
          extra=str(catalog_path))
    return Index.load(out)


def _doc(index, doc_id):
    return next(d for d in index.docs if d.id == doc_id)


def _dataset(index, code):
    return next(d for d in index.docs if d.kind == "dataset"
                and str(d.meta.get("short_code")).lower() == code)


# --------------------------------------------------------------------------- #
# Leitura do catálogo
# --------------------------------------------------------------------------- #
def test_variaveis_do_catalogo_sem_qc(catalog):
    info = variable_info(catalog)
    assert "qc_rh" not in info
    assert info["temp"]["long_name"] == "Air temperature"
    assert units_text(info["temp"]["units"]) == "degC"   # só a versão vigente
    # variável de texto: o "units" é ignorado (mesma regra do ACT)
    assert info["station_id"]["long_name"] == "Station identifier"
    assert units_text(info["station_id"]["units"]) is None
    assert info["reflectivity"]["procs"] == ["kazrge.a1"]
    assert info["rh"]["classes"] == {"1twrmr"}


def test_nivel_preferido_e_lacunas():
    c = {"levels": {
        "a1": {"variables": [{"name": "x", "long_name": "raw x", "units": "V"}]},
        "c1": {"variables": [{"name": "x", "long_name": "X value", "units": ""},
                             {"name": "qc_x", "is_qc": True}]}}}
    v = class_variables(c)
    assert set(v) == {"x"}
    assert v["x"]["long_name"] == "X value"      # c1 vem antes de a1
    assert v["x"]["units"] == "V"                # lacuna preenchida pelo a1
    assert v["x"]["levels"] == ["c1", "a1"]


def test_cada_produto_conta_uma_vez():
    def cls(long_name, unit, levels):
        return {"levels": {lv: {"variables": [
            {"name": "temp", "long_name": long_name, "units": unit}]}
            for lv in levels}}
    cat = {"classes": {"a": cls("Air temperature", "degC", ["a1", "b1", "c1"]),
                       "b": cls("Temperature", "K", ["b1"]),
                       "c": cls("Temperature", "K", ["b1"])}}
    info = variable_info(cat)["temp"]
    assert info["long_name"] == "Temperature"    # 2 produtos x 1
    assert info["units"] == Counter({"K": 2, "degC": 1})


def test_unidades_alternativas():
    assert unit_split(Counter({"degC": 3, "K": 1})) == ("degC", ["K"])
    assert unit_split(Counter({"degC": 9, "K": 1})) == ("degC", [])
    assert units_text(Counter({"degC": 3, "K": 1})) == "degC (também: K)"
    assert units_text(Counter()) is None
    assert display_unit(" 1 ") == "1 (adimensional)"
    assert display_unit("") is None


def test_dqr_de_varios_datastreams_vira_um_so():
    base = {"id": "D1", "category": "suspect", "dates": []}
    merged = merge_dqrs([
        {**base, "datastream": "sgpmetE13.b1", "class": "met"},
        {**base, "datastream": "sgpmetE15.b1", "class": "met"},
        {**base, "datastream": "sgpmetE13.b1", "class": "met"},
        {"id": "D0", "category": "incorrect", "dates": [],
         "datastream": "enametC1.b1", "class": "met"}])
    assert [m["id"] for m in merged] == ["D0", "D1"]
    assert merged[1]["datastreams"] == ["sgpmetE13.b1", "sgpmetE15.b1"]
    assert merged[1]["classes"] == ["met"]


def test_sem_catalogo_nada_muda():
    docs = ["qualquer"]
    assert apply_extra(docs, None) is docs
    assert apply_extra(docs, {}) is docs


def test_load_extra(tmp_path, monkeypatch, catalog_path):
    assert load_extra("none") == (None, None)
    assert load_extra(str(tmp_path / "nao_existe.json")) == (None, None)
    data, used = load_extra(str(catalog_path))
    assert data["version"] == 1 and used == str(catalog_path)
    monkeypatch.chdir(tmp_path)
    assert load_extra("auto") == (None, None)           # sem catálogo local
    (tmp_path / "data" / "arm_extra").mkdir(parents=True)
    save_json({"version": 1}, tmp_path / "data/arm_extra/catalog.json")
    assert load_extra("auto")[0] == {"version": 1}


def test_indice_padrao_dos_testes_nao_usa_catalogo(index):
    assert not [d for d in index.docs if d.kind in ("dqr", "site")]


# --------------------------------------------------------------------------- #
# Fichas enriquecidas
# --------------------------------------------------------------------------- #
def test_ficha_do_dataset_ganha_dod_datastream_e_dqrs(index_extra):
    d = _dataset(index_extra, "1twrmr")
    core = d.lines[:d.meta["n_core"]]
    assert any(x.startswith("Níveis de processamento (DOD oficial): c1 "
                            "(versão 1.1") for x in core)
    assert "Datastreams confirmados: sgp1twrmrC1.c1 (DOI 10.5439/0000001)" in core
    dqr = next(x for x in core if x.startswith("Relatórios de qualidade"))
    assert "2 (" in dqr and dqr.index("D200505.2") < dqr.index("D190101.1")
    assert "sgp1twrmrC1.c1" in d.keys
    units = {v["name"]: v.get("units") for v in d.meta["variables"]}
    assert units["rh"] == "%" and units["mixing_ratio"] == "g/kg"
    assert "station_id" in units and units["station_id"] is None  # só no DOD
    assert "qc_rh" not in units
    # o produto 30-min não tem DOD no servidor falso: fica intacto
    d30 = _dataset(index_extra, "30twrmr")
    assert not any("DOD oficial" in x for x in d30.lines)


def test_fichas_de_variavel(index_extra):
    refl = next(d for d in index_extra.docs if d.kind == "variable"
                and d.meta.get("variable", "").lower() == "reflectivity")
    assert "Unidade (DOD oficial): dBZ" in refl.lines
    sid = _doc(index_extra, "var:station_id")
    assert sid.meta["source"] == "dod"
    assert "Nome longo (DOD oficial): Station identifier" in sid.lines


def test_ficha_de_dqr(index_extra):
    d = _doc(index_extra, "dqr:D190101.1")
    assert d.kind == "dqr"
    text = "\n".join(d.lines)
    assert "Categoria: Suspect — dados suspeitos" in text
    assert "Variáveis afetadas: rh, mixing_ratio" in text
    assert "sítio SGP, facility C1, nível c1" in text
    assert "Período afetado: 2019-01-01" in text


def test_ficha_de_sitio(index_extra):
    text = "\n".join(_doc(index_extra, "site:sgp").lines)
    assert "C1 (latitude 36.607322, longitude -97.487643)" in text
    assert "E13 (coordenadas não informadas)" in text


# --------------------------------------------------------------------------- #
# Busca
# --------------------------------------------------------------------------- #
def test_pergunta_de_qualidade_traz_dqrs_mais_recentes_primeiro(index_extra):
    hits = index_extra.search("Há problemas de qualidade conhecidos no 1twrmr?",
                              k=5)
    ids = [h.doc.id for h in hits]
    assert hits[0].doc.kind == "dataset"
    assert ids[1:3] == ["dqr:D200505.2", "dqr:D190101.1"]


def test_sem_intencao_de_qualidade_sem_dqrs(index_extra):
    hits = index_extra.search("Qual o DOI do 1twrmr?", k=5)
    assert not [h for h in hits if h.doc.kind == "dqr"]


def test_dqr_pelo_identificador(index_extra):
    hits = index_extra.search("O que diz o DQR D190101.1?", k=3)
    assert hits[0].doc.id == "dqr:D190101.1"


def test_datastream_confirmado_acha_o_produto_certo(index_extra):
    hits = index_extra.search("Qual o DOI do datastream sgp1twrmrC1.c1?", k=3)
    assert str(hits[0].doc.meta.get("short_code")).lower() == "1twrmr"
    assert all(str(h.doc.meta.get("short_code")).lower() != "30twrmr"
               for h in hits)


def test_coordenadas_do_sitio(index_extra):
    hits = index_extra.search("Quais as coordenadas das facilities do sítio SGP?",
                              k=3)
    assert hits[0].doc.id == "site:sgp"


def test_unidade_aparece_no_contexto(index_extra):
    q = "Qual a unidade de rh no 1twrmr?"
    ctx, used = build_context(index_extra.search(q, k=3), q, max_chars=5000)
    assert "Variável rh: Relative humidity (unidade: %)" in ctx
