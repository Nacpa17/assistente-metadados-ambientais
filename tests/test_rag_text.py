from assistente_metadados.rag.text import contains_phrase, doi_suffix, phrase, tokenize


def test_identificador_composto_gera_inteiro_e_partes():
    toks = tokenize("O que é mean_doppler_velocity?")
    assert "mean_doppler_velocity" in toks
    assert {"mean", "doppler", "velocity"} <= set(toks)


def test_doi_e_codigo_preservados():
    toks = tokenize("Qual dataset tem o DOI 10.5439/1025256? E o 1twrmr?")
    assert "10.5439/1025256" in toks and "1025256" in toks
    assert "1twrmr" in toks          # sem quebrar em "1" + "twrmr"


def test_acentos_e_stopwords():
    toks = tokenize("Qual é o período de coleta do dataset?")
    assert "periodo" in toks and "coleta" in toks
    assert not {"qual", "e", "o", "de", "do", "dataset"} & set(toks)


def test_sigla_em_maiusculas_nao_vira_stopword():
    # "AOS" (Aerosol Observing System) x "aos" (preposição)
    assert "aos" in tokenize("O que significa a sigla AOS?")
    assert "aos" not in tokenize("Quais datasets pertencem aos observatórios?")


def test_phrase_respeita_bordas_de_palavra():
    q = phrase("Qual o período do 1twrmr?")
    assert contains_phrase(q, "1twrmr")
    assert not contains_phrase(q, "twrmr")       # não casa "pedaço" do código
    assert not contains_phrase(phrase("dados do 30twrmr"), "1twrmr")


def test_doi_suffix():
    assert doi_suffix("https://doi.org/10.5439/1025256") == "10.5439/1025256"
    assert doi_suffix("sem doi") is None
    assert doi_suffix(None) is None
