"""Coleta nos serviços oficiais da ARM (com servidor falso nos formatos reais)."""

import json

import pytest

from fake_arm import FakeArmSession, FakeResponse
from src.collect.arm_extra import build_catalog, collect, corpus_targets
from src.collect.arm_services import (
    ArmClient, ArmServiceError, doi_from_text, fetch_citation, fetch_dod,
    fetch_dqrs, fetch_facilities, parse_dod, parse_dqrs,
)
from src.common.arm_names import parse_datastream


def client(tmp_path=None, **kw):
    return ArmClient(cache_dir=tmp_path, delay_s=0, session=kw.pop("session", None)
                     or FakeArmSession(), **kw)


@pytest.mark.parametrize("name,parts", [
    ("sgpmetE13.b1", ("sgp", "met", "E13", "b1")),
    ("sgp30baebbrE39.c1", ("sgp", "30baebbr", "E39", "c1")),
    ("maoarmbeatmM1.c1", ("mao", "armbeatm", "M1", "c1")),
    ("maomfrsraod1michM1.s1", ("mao", "mfrsraod1mich", "M1", "s1")),
])
def test_nome_de_datastream(name, parts):
    d = parse_datastream(name)
    assert (d["site"], d["class"], d["facility"], d["level"]) == parts
    assert d["proc"] == f"{parts[1]}.{parts[3]}"


@pytest.mark.parametrize("bad", ["", "met", "sgpmet.b1", "SGPmetE13.b1",
                                 "sgpmetE13"])
def test_nome_invalido(bad):
    assert parse_datastream(bad) is None


def test_dod_usa_versao_mais_recente_e_regras_do_act():
    c = client()
    dod = parse_dod(fetch_dod(c, "1twrmr.c1"), "1twrmr.c1")
    assert dod["version"] == "1.1" and dod["n_versions"] == 2
    by = {v["name"]: v for v in dod["variables"]}
    assert by["rh"]["units"] == "%" and by["rh"]["long_name"] == "Relative humidity"
    assert by["temp"]["units"] == "degC"                  # da versão 1.1
    assert by["qc_rh"]["is_qc"] and not by["rh"]["is_qc"]
    assert by["station_id"]["units"] is None              # variável de texto
    assert dod["global_atts"] == {"datastream": "sgp1twrmrC1.c1",
                                  "doi": "10.5439/0000001",
                                  "command_line": "twrmr -s sgp"}
    assert parse_dod(fetch_dod(c, "naoexiste.b1"), "naoexiste.b1") is None


def test_dqrs():
    c = client()
    dqrs = parse_dqrs(fetch_dqrs(c, "sgp1twrmrC1.c1", "19980401", "20261022"),
                      "sgp1twrmrC1.c1")
    assert [d["id"] for d in dqrs] == ["D190101.1", "D200505.2"]
    d0 = dqrs[0]
    assert d0["category"] == "suspect" and d0["variables"] == ["rh", "mixing_ratio"]
    assert d0["dates"][0]["start"].startswith("2019-01-01")
    assert d0["url"].endswith("dqrid=D190101.1")
    url = c.session.calls[-1][1]
    assert url.endswith("/sgp1twrmrC1.c1/19980401/20261022/incorrect,suspect,missing")
    assert parse_dqrs(fetch_dqrs(c, "sgpxyzC1.b1", "1", "2"), "sgpxyzC1.b1") == []


def test_citacao_e_doi():
    c = client()
    cit = fetch_citation(c, "sgp1twrmrC1.c1", "1998-04-01", "2026-10-22")
    assert "Sixty Meter Tower" in cit and doi_from_text(cit) == "10.5439/0000001"
    assert fetch_citation(c, "naoexisteC1.b1", "2000-01-01", "2001-01-01") is None


def test_sitios_com_a_consulta_do_act():
    c = client()
    fac = fetch_facilities(c, "sgp")
    assert fac[0] == {"site_code": "sgp", "facility_code": "C1",
                      "site_type": "Fixed", "latitude": 36.607322,
                      "longitude": -97.487643}
    assert fac[1]["latitude"] is None                     # sem coordenadas


def test_cache_evita_nova_requisicao(tmp_path):
    s = FakeArmSession()
    c1 = client(tmp_path, session=s)
    fetch_dod(c1, "1twrmr.c1")
    fetch_dod(c1, "naoexiste.b1")                         # 404 também em cache
    assert len(s.calls) == 2
    c2 = client(tmp_path, session=s)
    assert fetch_dod(c2, "1twrmr.c1") is not None
    assert fetch_dod(c2, "naoexiste.b1") is None
    assert len(s.calls) == 2 and c2.n_cache_hits == 2
    c3 = client(tmp_path, session=s, refresh=True)        # --atualizar
    fetch_dod(c3, "1twrmr.c1")
    assert len(s.calls) == 3


def test_retenta_erros_temporarios(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    url = "https://pcm.arm.gov/pcm/api/dods/kazrge.a1"
    s = FakeArmSession({url: [FakeResponse(503, text="x"),
                              FakeResponse(429, text="x")]})
    c = client(session=s)
    assert parse_dod(fetch_dod(c, "kazrge.a1"), "kazrge.a1")["version"] == "2.0"
    assert len(s.calls) == 3


def test_erros_definitivos(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    url = "https://pcm.arm.gov/pcm/api/dods/kazrge.a1"
    c = client(session=FakeArmSession({url: [FakeResponse(403, text="no")]}))
    with pytest.raises(ArmServiceError, match="403"):
        fetch_dod(c, "kazrge.a1")
    c = client(session=FakeArmSession(
        {url: [FakeResponse(200, text="<!DOCTYPE html><p>erro</p>")]}))
    with pytest.raises(ArmServiceError, match="não-JSON"):
        fetch_dod(c, "kazrge.a1")
    c = client(session=FakeArmSession({url: [FakeResponse(503, text="x")] * 3}),
               retries=3)
    with pytest.raises(ArmServiceError, match="Esgotadas"):
        fetch_dod(c, "kazrge.a1")


def test_coleta_completa_a_partir_do_corpus(corpus_dir, tmp_path):
    targets = corpus_targets(corpus_dir)
    assert {"1twrmr", "30twrmr", "kazrge", "ldquants"} <= set(targets)
    assert targets["1twrmr"]["pairs"] == {("sgp", "C1")}
    assert targets["1twrmr"]["end"] == "ongoing"
    assert targets["kazrge"]["pairs"] == set()            # sem citação/site

    s = FakeArmSession()
    explicit = [parse_datastream("nsakazrgeC1.a1")]
    res = collect(targets, explicit, client(tmp_path, session=s),
                  levels=["b1", "c1", "a1"], do_dods=True, do_dqrs=True,
                  do_citations=True, do_sites=True)
    cat = build_catalog(res, {"teste": True})

    assert set(cat["classes"]["1twrmr"]["levels"]) == {"c1"}
    assert set(cat["classes"]["kazrge"]["levels"]) == {"a1"}  # nível explícito
    ds = {d["datastream"]: d for c in cat["classes"].values()
          for d in c["datastreams"]}
    assert ds["sgp1twrmrC1.c1"]["confirmed"] is True
    assert ds["sgp1twrmrC1.c1"]["doi"] == "10.5439/0000001"
    assert ds["sgp1twrmrC1.c1"]["dqr_counts"] == {"suspect": 1, "incorrect": 1}
    assert ds["nsakazrgeC1.a1"]["confirmed"] is True       # informado
    assert cat["stats"]["dqrs"] == 2
    assert {d["class"] for d in cat["dqrs"]} == {"1twrmr"}
    assert cat["stats"]["facilities"] == 2
    assert cat["stats"]["variables_with_units"] >= 4
    json.dumps(cat)                                       # serializável
    # nenhum nível foi testado para classes sem DOD além dos pedidos
    tried = [u for m, u in s.calls if "/dods/30twrmr." in u]
    assert len(tried) == 3


def test_datastream_montado_sem_evidencia_nao_e_confirmado(corpus_dir, tmp_path):
    targets = corpus_targets(corpus_dir)
    s = FakeArmSession()
    res = collect({"30twrmr": targets["30twrmr"]}, [],
                  client(tmp_path, session=s), levels=["c1"],
                  do_dods=False, do_dqrs=True, do_citations=False,
                  do_sites=False)
    entry = res["classes"]["30twrmr"]["datastreams"]
    assert entry == [] or all(not d["confirmed"] for d in entry)


def test_sem_rede_interrompe_cedo(corpus_dir, tmp_path):
    from src.collect.arm_extra import ArmUnreachable
    blocked = FakeArmSession({"https://": [FakeResponse(403, text="")] * 100})
    with pytest.raises(ArmUnreachable, match="5 consultas seguidas"):
        collect(corpus_targets(corpus_dir), [], client(tmp_path, session=blocked),
                levels=["b1", "c1", "a1"], do_dods=True, do_dqrs=True,
                do_citations=False, do_sites=True, max_consecutive_errors=5)
    assert len(blocked.calls) == 5          # parou na 5ª, sem insistir


def test_erros_esparsos_nao_interrompem(corpus_dir, tmp_path):
    # 403 só no 1º DOD; o resto responde normalmente
    s = FakeArmSession({"https://pcm.arm.gov/pcm/api/dods/1twrmr.": [
        FakeResponse(403, text="")]})
    res = collect(corpus_targets(corpus_dir), [], client(tmp_path, session=s),
                  levels=["c1"], do_dods=True, do_dqrs=False,
                  do_citations=False, do_sites=False, max_consecutive_errors=2)
    assert "1twrmr" in res["classes"]
