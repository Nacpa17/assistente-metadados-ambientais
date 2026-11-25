"""Streaming: quebras de linha da resposta não podem partir o evento SSE."""

from assistente_metadados.api.sse import parse_sse, sse_event


def test_texto_com_quebras_de_linha_fica_inteiro():
    pieces = ["Datasets:\n", "- 1twrmr\n\n", "- 30twrmr", ' com "aspas"']
    stream = "".join(sse_event(p) for p in pieces)
    # cada evento é UMA linha de dados: o separador "\n\n" só aparece entre
    # eventos, nunca dentro do texto
    assert stream.count("\n\n") == len(pieces)
    got = [d for _, d in parse_sse(stream)]
    assert "".join(got) == "".join(pieces)


def test_eventos_nomeados():
    sources = [{"n": 1, "title": "Ficha", "doi": "https://doi.org/10.5439/1"}]
    stream = (sse_event(sources, event="sources") + sse_event("oi")
              + sse_event("[END]", event="done"))
    assert parse_sse(stream) == [("sources", sources), (None, "oi"),
                                 ("done", "[END]")]


def test_acentos_preservados():
    assert "período" in sse_event("período")
