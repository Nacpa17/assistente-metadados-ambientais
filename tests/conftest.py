"""Fixtures compartilhadas: mini-corpus sintético + índice RAG montado nele."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (ROOT, TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fixtures_arm import write_corpus  # noqa: E402

CURATED_GLOSSARY = ROOT / "data" / "glossary" / "glossary.yaml"
EXPANSIONS = ROOT / "data" / "rag" / "expansoes_pt_en.yaml"


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    return write_corpus(tmp_path_factory.mktemp("arm") / "raw_json")


@pytest.fixture(scope="session")
def index_path(corpus_dir, tmp_path_factory) -> Path:
    from assistente_metadados.rag.build_index import build
    out = tmp_path_factory.mktemp("rag") / "index.json"
    # extra="none": os testes não podem depender de data/arm_extra local
    build(str(corpus_dir), str(CURATED_GLOSSARY), str(EXPANSIONS), str(out),
          extra="none")
    return out


@pytest.fixture(scope="session")
def index(index_path):
    from assistente_metadados.rag.index import Index
    return Index.load(index_path)


@pytest.fixture(scope="session")
def catalog(corpus_dir, tmp_path_factory):
    """Catálogo de serviços ARM coletado do servidor FALSO (tests/fake_arm.py:
    formatos reais do ACT, dados inventados) para o mini-corpus."""
    from fake_arm import FakeArmSession
    from assistente_metadados.collect.arm_extra import build_catalog, collect, corpus_targets
    from assistente_metadados.collect.arm_services import ArmClient
    from assistente_metadados.common.arm_names import parse_datastream
    client = ArmClient(cache_dir=tmp_path_factory.mktemp("arm_cache"),
                       delay_s=0, session=FakeArmSession())
    res = collect(corpus_targets(corpus_dir),
                  [parse_datastream("nsakazrgeC1.a1")], client,
                  levels=["b1", "c1", "a1"], do_dods=True, do_dqrs=True,
                  do_citations=True, do_sites=True)
    return build_catalog(res, {"teste": True})


@pytest.fixture(scope="session")
def catalog_path(catalog, tmp_path_factory) -> Path:
    from assistente_metadados.common.io import save_json
    p = tmp_path_factory.mktemp("arm_extra") / "catalog.json"
    save_json(catalog, p)
    return p
