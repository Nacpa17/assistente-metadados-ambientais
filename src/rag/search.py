"""Busca no índice RAG pelo terminal — para conferir o que o modelo vai "ler".

Uso:
    python -m src.rag.search "Qual o DOI de 1twrmr?"
    python -m src.rag.search "O que é reflectivity?" -k 5 --contexto
    python -m src.rag.search            # modo interativo (Ctrl+C para sair)
"""

from __future__ import annotations

import argparse

from src.rag.context import build_context
from src.rag.index import Index


def show(index: Index, query: str, k: int, with_context: bool,
         kind: str | None, max_chars: int) -> None:
    hits = index.search(query, k=k, kinds=[kind] if kind else None)
    print(f"\nPergunta: {query}")
    if not hits:
        print("  (nenhum documento encontrado)")
        return
    print(f"  {'#':>2}  {'tipo':<9} {'BM25':>7}  {'exato':<5}  documento")
    for n, h in enumerate(hits, 1):
        exact = "sim" if h.boost else ""
        print(f"  {n:>2}  {h.doc.kind:<9} {h.score:7.2f}  {exact:<5}  "
              f"{h.doc.id} — {h.doc.title[:70]}")
    if with_context:
        ctx, used = build_context(hits, query, max_chars=max_chars)
        print(f"\n--- contexto que iria para o prompt ({len(used)} ficha(s), "
              f"{len(ctx)} caracteres) ---\n{ctx}\n---")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="pergunta (vazio = interativo)")
    ap.add_argument("--index", default="data/rag/index.json")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--contexto", "--context", dest="context",
                    action="store_true", help="mostra o contexto montado")
    ap.add_argument("--tipo", "--kind", dest="kind", default=None,
                    choices=["dataset", "glossary", "variable", "facility",
                             "corpus", "dqr", "site"])
    ap.add_argument("--max-chars", type=int, default=5000)
    a = ap.parse_args()

    index = Index.load(a.index)
    print(f"Índice: {a.index} | {index.info.get('n_docs', len(index.docs))} "
          f"documentos {index.info.get('n_docs_by_kind', '')}")
    if a.query:
        show(index, " ".join(a.query), a.k, a.context, a.kind, a.max_chars)
        return
    print("Modo interativo — digite uma pergunta (Ctrl+C para sair).")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                show(index, q, a.k, a.context, a.kind, a.max_chars)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
