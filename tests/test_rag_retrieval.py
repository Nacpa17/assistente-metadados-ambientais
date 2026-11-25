"""Recuperação: a ficha certa vem em 1º lugar e distratores ficam de fora."""

import pytest

from assistente_metadados.rag.index import Index

TWR1 = ("Sixty Meter Tower: mixing ratio at surface, 25-m, and 60-m, "
        "1-min average - 1twrmr")
TWR30 = ("Sixty Meter Tower: mixing ratio at surface, 25-m, and 60-m, "
         "30-min average - 30twrmr")


def ids(hits):
    return [h.doc.id for h in hits]


@pytest.mark.parametrize("query,expected", [
    ("Qual o período de coleta do dataset 1twrmr?", "ds:1twrmr"),
    (f"Qual o DOI de {TWR30}?", "ds:30twrmr"),
    (f"Onde posso acessar {TWR1}?", "ds:1twrmr"),
    ("Qual dataset tem o DOI 10.5439/0000004?", "ds:mwr3c"),
    ("O que é mean_doppler_velocity?", "gl:mean_doppler_velocity"),
    ("O que significa a sigla KAZR?", "gl:kazr"),
    ("O que significa a sigla AOS?", "gl:aos"),
    ("Quais datasets medem reflectivity?", "var:reflectivity"),
    ("Quantos datasets medem a variável rh?", "var:rh"),
    ("Qual variável é a mais frequentemente medida nos datasets do corpus "
     "ARM analisado?", "corpus"),
])
def test_top1(index, query, expected):
    assert ids(index.search(query, k=3))[0] == expected


def test_titulos_quase_iguais_nao_entram_como_distratores(index):
    # 1twrmr e 30twrmr têm o mesmo título e DOIs diferentes: com a pergunta
    # citando um deles, o outro NÃO pode ir para o contexto.
    got = ids(index.search(f"Qual o DOI de {TWR30}?", k=5))
    assert got[0] == "ds:30twrmr"
    assert "ds:1twrmr" not in got
    got = ids(index.search("Qual o período de coleta do dataset 1twrmr?", k=5))
    assert "ds:30twrmr" not in got


def test_multi_hop_traz_os_dois(index):
    got = ids(index.search(
        "Entre os datasets 1twrmr e kazrge, qual cobre o período mais longo?",
        k=3))
    assert set(got[:2]) == {"ds:1twrmr", "ds:kazrge"}


def test_dataset_citado_vem_antes_da_variavel_e_do_glossario(index):
    got = ids(index.search("O que significa a variável 'rh' no dataset 1twrmr?",
                           k=3))
    assert got[0] == "ds:1twrmr"
    assert {"var:rh", "gl:rh"} <= set(got)


def test_intencao_define_glossario_ou_indice(index):
    assert ids(index.search("O que é reflectivity?", k=2))[0] == "gl:reflectivity"
    assert ids(index.search("Quais datasets medem reflectivity?",
                            k=2))[0] == "var:reflectivity"


def test_preposicao_aos_nao_ativa_sigla_aos(index):
    # A ficha do AOS pode até aparecer por outra palavra ("observatórios"
    # está na definição dele), mas NUNCA como casamento exato da preposição.
    got = index.search("Quais datasets pertencem aos observatórios do SGP?", k=5)
    assert got[0].doc.id == "fac:southern great plains (sgp)"
    assert not any("aos" in [k.lower() for k in h.matched_keys] for h in got)
    assert all(h.boost == 0 for h in got if h.doc.id == "gl:aos")


def test_fora_de_escopo_sem_casamento_exato(index):
    for q in ("Qual a cotação do dólar hoje?",
              "Qual a previsão do tempo para amanhã em Brasília?"):
        assert not any(h.boost for h in index.search(q, k=5))


def test_expansao_pt_en_liga_umidade_a_humidity(index):
    q = "Preciso de dados de umidade relativa medidos na torre"
    with_exp = ids(index.search(q, k=5, expand=True))
    without = ids(index.search(q, k=5, expand=False))
    assert "ds:1twrmr" in with_exp or "ds:30twrmr" in with_exp
    assert not ({"ds:1twrmr", "ds:30twrmr"} & set(without))


def test_salvar_e_carregar_preserva_resultados(index, tmp_path):
    p = tmp_path / "idx.json"
    index.save(p)
    again = Index.load(p)
    for q in ("Qual o DOI do 1twrmr?", "O que é KAZR?",
              "dados de umidade em Southern Great Plains"):
        assert ids(again.search(q, k=5)) == ids(index.search(q, k=5))


def test_indice_ausente_da_mensagem_util(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        Index.load(tmp_path / "nao_existe.json")


def test_registro_sem_campos_nao_quebra(index):
    # ldquants não tem descrição, coordenadas nem facility
    got = index.search("Qual o formato dos dados de ldquants?", k=1)
    assert got and got[0].doc.id == "ds:ldquants"
