"""Contexto e prompt: o modelo precisa receber os fatos EXATOS, no formato
que já aprendeu, sem estourar o limite de tokens."""

import json

from assistente_metadados.common.prompts import (DEFAULT_RAG_INSTRUCTIONS, NO_CONTEXT_TEXT,
                                build_chat_prompt, build_rag_prompt,
                                build_training_prompt)
from assistente_metadados.prompts.generator import parse_record
from assistente_metadados.rag.context import build_context, fit_to_budget, render_doc
from assistente_metadados.rag.pipeline import RagPipeline, RagSettings, sources_payload

SYSTEM = "Você é um assistente técnico."


def test_ficha_usa_as_mesmas_frases_do_gerador(index, corpus_dir):
    rec = parse_record(json.loads((corpus_dir / "1twrmr.json")
                                  .read_text(encoding="utf-8")))
    card = "\n".join(render_doc(index.by_id["ds:1twrmr"]))
    # exatamente as strings que viram respostas no dataset de treino/gold
    assert rec.period_display == "1998-04-01 até o presente (coleta contínua)"
    assert f"Período de coleta: {rec.period_display}" in card
    assert "Latitude: 36.607322, Longitude: -97.487643" in card
    assert ("Observatório (facility): Southern Great Plains (SGP), Central "
            "Facility, Lamont, OK (C1)") in card
    assert f"DOI / acesso: {rec.doi}" in card
    assert "Formato dos dados: NetCDF" in card


def test_variavel_citada_sobe_e_ganha_descricao(index):
    doc = index.by_id["ds:1twrmr"]
    lines = render_doc(doc, "O que significa a variável 'rh' no dataset 1twrmr?")
    var_line = next(l for l in lines if l.startswith("Variáveis medidas"))
    assert var_line.split(": ", 1)[1].startswith("rh")
    assert "Variável rh: Relative humidity" in lines
    # sem citar variável: ordem original, sem descrições extras
    plain = render_doc(doc, "Qual o DOI do 1twrmr?")
    assert not any(l.startswith("Variável ") for l in plain)


def test_contexto_respeita_orcamento_e_numera(index):
    hits = index.search("Quais datasets medem reflectivity?", k=5)
    ctx, used = build_context(hits, "x", max_chars=600)
    assert ctx.startswith("[1] ")
    assert len(used) >= 1
    assert len(ctx) <= 600 + 1
    ctx_all, used_all = build_context(hits, "x", max_chars=100_000)
    assert len(used_all) == len(hits)
    assert "[2] " in ctx_all


def test_fit_to_budget_remove_fichas_do_fim():
    calls = []

    def make(hs):
        calls.append(len(hs))
        return "x" * (100 * len(hs) + 10), list(hs)

    prompt, used = fit_to_budget(make, ["a", "b", "c"], len, max_tokens=250)
    assert used == ["a", "b"] and len(prompt) <= 250
    assert calls == [3, 2]
    # nem sem fichas cabe: devolve a versão sem contexto
    prompt, used = fit_to_budget(make, ["a"], len, max_tokens=5)
    assert used == []


def test_prompt_sem_rag_continua_igual_ao_do_treino():
    # regressão: o caminho antigo não pode mudar (o LoRA foi treinado nele)
    assert build_chat_prompt(SYSTEM, [], "Pergunta X?") == \
        build_training_prompt(SYSTEM, "Pergunta X?")


def test_prompt_rag_mantem_formato_e_poe_contexto_antes_da_pergunta():
    hist = [{"role": "user", "content": "Oi?"},
            {"role": "assistant", "content": "Olá."}]
    p = build_rag_prompt(SYSTEM, hist, "[1] Ficha de dataset\nDOI: X",
                         "Qual o DOI?")
    assert p.startswith("<<SYS>>\n" + SYSTEM)
    assert DEFAULT_RAG_INSTRUCTIONS in p
    assert p.endswith("Pergunta: Qual o DOI?\nResposta:")
    assert p.index("Resposta: Olá.") < p.index("Contexto (metadados recuperados)") \
        < p.index("Pergunta: Qual o DOI?")
    assert NO_CONTEXT_TEXT in build_rag_prompt(SYSTEM, [], "", "Q?")


def test_pipeline_monta_prompt_com_fontes(index_path):
    rag = RagPipeline.from_config({"rag": {"index_path": str(index_path),
                                           "top_k": 3}})
    res = rag.build_prompt(SYSTEM, "Qual o DOI do dataset 1twrmr?")
    assert "DOI / acesso: https://doi.org/10.5439/0000001" in res.prompt
    assert res.sources[0]["id"] == "ds:1twrmr"
    assert res.sources[0]["doi"] == "https://doi.org/10.5439/0000001"
    assert res.sources[0]["exact_match"] is True
    json.dumps(res.sources)                       # serializável para a API


def test_pipeline_respeita_limite_de_tokens(index_path):
    rag = RagPipeline.from_config(
        {"rag": {"index_path": str(index_path), "top_k": 5,
                 "max_input_tokens": 700}})
    count = lambda s: len(s) // 3                 # "tokenizer" aproximado
    res = rag.build_prompt(SYSTEM, "Quais datasets medem reflectivity?",
                           count_tokens=count)
    assert count(res.prompt) <= 700
    assert len(res.hits) < len(res.retrieved)


def test_pergunta_de_continuacao_usa_a_pergunta_anterior(index_path):
    rag = RagPipeline.from_config({"rag": {"index_path": str(index_path)}})
    hist = [{"role": "user", "content": "Me fale do dataset kazrge."},
            {"role": "assistant", "content": "É um radar."}]
    # "DOI" casa com o glossário, mas não diz DE QUAL dataset: usa o anterior
    res = rag.build_prompt(SYSTEM, "E qual o DOI dele?", history=hist)
    assert res.hits and res.hits[0].doc.id == "ds:kazrge"
    # pergunta que já cita outro dataset NÃO herda o anterior
    res = rag.build_prompt(SYSTEM, "Qual o DOI do 1twrmr?", history=hist)
    assert [h.doc.id for h in res.hits if h.doc.kind == "dataset"] == ["ds:1twrmr"]
    # continuação de pergunta de glossário
    hist2 = [{"role": "user", "content": "O que é KAZR?"},
             {"role": "assistant", "content": "Um radar."}]
    res = rag.build_prompt(SYSTEM, "E para que serve?", history=hist2)
    assert res.hits[0].doc.id == "gl:kazr"


def test_settings_overrides_e_tipos():
    s = RagSettings.from_config({"rag": {"top_k": "4", "inexistente": 1}},
                                top_k=None, index_path="x.json")
    assert s.top_k == 4 and s.index_path == "x.json"


def test_sources_payload_vazio():
    assert sources_payload([]) == []
