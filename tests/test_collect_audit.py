"""Coletor (nomes únicos, reparo de colisões, procedência) e auditoria."""

import hashlib
import json
import sys

import pytest

from fixtures_arm import write_corpus
from src.collect import audit as audit_mod
from src.collect.arm_collector import (ORPHAN_DIR, ArmSourceConfig, State,
                                       collect_source, filename_for, slugify)
from src.common.io import load_json, save_json

SITEMAP = "https://adc.arm.gov/sitemap.xml"
URL_A = "https://adc.arm.gov/discovery/ds/meta.html"
URL_B = "https://adc.arm.gov/discovery/ds/metb.html"
# títulos longos que só diferem DEPOIS do 80º caractere do slug (dados fictícios)
LONG = ("Surface Meteorological Instrumentation (MET): temperature, humidity, "
        "pressure and wind, 1-min averages")
DS_A = {"@type": "Dataset", "name": f"{LONG}, variant A - meta",
        "url": "https://doi.org/10.5439/0000091"}
DS_B = {"@type": "Dataset", "name": f"{LONG}, variant B - metb",
        "url": "https://doi.org/10.5439/0000092"}


def old_filename(dataset):
    """Como a fase 2 nomeava: só o título truncado."""
    return f"{slugify(dataset['name'])}.json"


def page(dataset):
    return (f'<html><head><script type="application/ld+json">'
            f'{json.dumps(dataset)}</script></head><body></body></html>')


class Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text
        self.content = text.encode("utf-8")


class FakeHttp:
    def __init__(self, pages):
        self.pages = dict(pages)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url == SITEMAP:
            locs = "".join(f"<url><loc>{u}</loc></url>" for u in self.pages)
            return Resp(200, '<urlset xmlns="http://www.sitemaps.org/schemas/'
                             f'sitemap/0.9">{locs}</urlset>')
        if url in self.pages:
            return Resp(200, page(self.pages[url]))
        return Resp(404)


def cfg_for(tmp_path):
    return ArmSourceConfig(sitemap_url=SITEMAP, output_dir=str(tmp_path / "raw"),
                           state_file=str(tmp_path / "raw" / ".state.json"),
                           url_include_regex=r"/discovery/",
                           request_delay_seconds=0, max_retries=1)


def sha_of(dataset):
    return hashlib.sha256(json.dumps(dataset, ensure_ascii=False,
                                     sort_keys=True).encode("utf-8")).hexdigest()


def test_nome_unico_por_url():
    assert old_filename(DS_A) == old_filename(DS_B)        # o bug antigo
    fa, fb = filename_for(URL_A, DS_A), filename_for(URL_B, DS_B)
    assert fa != fb
    assert fa == filename_for(URL_A, DS_A)                 # determinístico
    assert fa.startswith("Surface_Meteorological") and fa.endswith(".json")
    assert filename_for(URL_A, {}).startswith("dataset__")


def test_coleta_nova_grava_procedencia_e_sha_do_original(tmp_path):
    cfg = cfg_for(tmp_path)
    c = collect_source(cfg, session=FakeHttp({URL_A: DS_A, URL_B: DS_B}))
    assert c["new"] == 2 and c["failed"] == 0
    st = State.load(tmp_path / "raw" / ".state.json")
    saved = load_json(tmp_path / "raw" / st.urls[URL_A]["filename"])
    assert saved["_fonte"]["url"] == URL_A
    assert saved["_fonte"]["coletor"] == "src.collect.arm_collector"
    assert st.urls[URL_A]["sha256"] == sha_of(DS_A)        # sem o _fonte
    # segunda rodada: nada mudou
    c2 = collect_source(cfg, session=FakeHttp({URL_A: DS_A, URL_B: DS_B}))
    assert c2["skipped"] == 2 and c2["new"] == c2["updated"] == 0


def test_conteudo_alterado_mantem_o_mesmo_arquivo(tmp_path):
    cfg = cfg_for(tmp_path)
    collect_source(cfg, session=FakeHttp({URL_A: DS_A}))
    before = State.load(tmp_path / "raw" / ".state.json").urls[URL_A]["filename"]
    changed = {**DS_A, "description": "nova descrição"}
    c = collect_source(cfg, session=FakeHttp({URL_A: changed}))
    st = State.load(tmp_path / "raw" / ".state.json")
    assert c["updated"] == 1 and st.urls[URL_A]["filename"] == before
    assert load_json(tmp_path / "raw" / before)["description"] == "nova descrição"


def _old_collision(tmp_path):
    """Estado e disco como o coletor antigo deixava: duas URLs, um arquivo
    (com o conteúdo da última gravada)."""
    raw = tmp_path / "raw"
    old = old_filename(DS_A)
    save_json(DS_B, raw / old)
    State(urls={URL_A: {"sha256": sha_of(DS_A), "filename": old, "ts": "x"},
                URL_B: {"sha256": sha_of(DS_B), "filename": old, "ts": "x"}}
          ).save(raw / ".state.json")
    return raw, old


def test_colisao_antiga_e_reparada(tmp_path):
    raw, old = _old_collision(tmp_path)
    c = collect_source(cfg_for(tmp_path),
                       session=FakeHttp({URL_A: DS_A, URL_B: DS_B}))
    assert c["repaired"] == 2
    st = State.load(raw / ".state.json")
    fa, fb = st.urls[URL_A]["filename"], st.urls[URL_B]["filename"]
    assert fa != fb and old not in (fa, fb)
    assert load_json(raw / fa)["name"] == DS_A["name"]     # registro recuperado
    assert load_json(raw / fb)["name"] == DS_B["name"]
    assert not (raw / old).exists()
    assert (raw / ORPHAN_DIR / old).exists()               # preservado, fora
    # o corpus não conta a pasta de órfãos
    rep = audit_mod.audit(raw)
    assert rep["n_valid"] == 2 and rep["state"]["n_overwritten_records"] == 0


def test_limite_nao_move_arquivo_ainda_referenciado(tmp_path):
    raw, old = _old_collision(tmp_path)
    c = collect_source(cfg_for(tmp_path),
                       session=FakeHttp({URL_A: DS_A, URL_B: DS_B}), limit=1)
    assert c["repaired"] == 1
    assert (raw / old).exists()        # URL_B ainda aponta para ele


# --------------------------------------------------------------------------- #
# Auditoria
# --------------------------------------------------------------------------- #
@pytest.fixture()
def messy_corpus(tmp_path):
    raw = write_corpus(tmp_path / "raw")
    (raw / "quebrado.json").write_text("{", encoding="utf-8")
    save_json({"sem": "nome"}, raw / "sem_nome.json")
    dup = load_json(raw / "1twrmr.json")
    save_json({**dup, "_fonte": {"url": "https://exemplo"}}, raw / "copia.json")
    # registro do Brasil (FICTÍCIO, só para o teste)
    save_json({"name": "Nephelometer, GoAmazon - aosnephmao",
               "url": "https://doi.org/10.5439/0000099",
               "about": [{"name": "Facility Information: ARM Mobile Facility "
                                  "(MAO) Manacapuru, Amazonas, Brazil"}]},
              raw / "brasil.json")
    save_json({"urls": {
        "u1": {"filename": "1twrmr.json"}, "u2": {"filename": "1twrmr.json"},
        "u3": {"filename": "sumiu.json"}}}, raw / ".state.json")
    return raw


def test_auditoria(messy_corpus):
    rep = audit_mod.audit(messy_corpus)
    assert rep["n_files"] == 10                 # .state.json não conta
    assert rep["n_valid"] == 8
    assert {i["file"] for i in rep["invalid"]} == {"quebrado.json",
                                                   "sem_nome.json"}
    assert rep["n_with_provenance"] == 1
    st = rep["state"]
    assert st["shared_files"] == {"1twrmr.json": ["u1", "u2"]}
    assert st["n_overwritten_records"] == 1
    assert st["missing_files"] == ["sumiu.json"]
    assert "kazrge.json" in st["files_not_in_state"]
    dups = rep["duplicates"]
    assert {"1twrmr.json", "copia.json"} in [set(g["files"]) for g in dups["doi"]]
    assert {"1twrmr.json", "copia.json"} in [set(g["files"])
                                             for g in dups["content"]]
    assert [b["short_code"] for b in rep["brazil"]] == ["aosnephmao"]
    assert rep["by_site"]["SGP"] == 4           # 1twrmr, 30twrmr, mwr3c, cópia
    assert rep["by_site"]["MAO"] == 1
    assert rep["coverage"]["coordenadas"]["n"] == 6
    rec = audit_mod.recommendations(rep)
    assert any("sobrescritos" in r for r in rec)
    assert not any("Nenhum registro do Brasil" in r for r in rec)


def test_auditoria_modo_estrito(messy_corpus, tmp_path, monkeypatch, capsys):
    out = tmp_path / "rel.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--json-dir", str(messy_corpus),
                                      "--out", str(out), "--estrito"])
    with pytest.raises(SystemExit) as e:
        audit_mod.main()
    assert e.value.code == 1
    assert load_json(out)["recommendations"]
    assert "Auditoria do corpus" in capsys.readouterr().out


def test_auditoria_corpus_limpo(tmp_path, monkeypatch):
    raw = write_corpus(tmp_path / "raw")
    rep = audit_mod.audit(raw)
    assert rep["invalid"] == [] and rep["state"] == {"present": False}
    assert not rep["duplicates"]["content"]
    monkeypatch.setattr(sys, "argv", ["audit", "--json-dir", str(raw),
                                      "--out", str(tmp_path / "r.json"),
                                      "--estrito"])
    audit_mod.main()                            # não sai com erro
