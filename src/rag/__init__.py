"""Camada de recuperação (RAG) sobre os metadados ARM — fase 3 do PIBIC.

Ideia: em vez de exigir que o modelo MEMORIZE DOIs, datas e coordenadas
(o teto do fine-tuning identificado na fase 2), buscamos a ficha certa nos
JSON-LD e a colocamos no prompt para o modelo apenas LER a informação.

Tudo aqui é Python puro (sem torch/transformers), então roda em qualquer
máquina, inclusive sem GPU:

    python -m src.rag.build_index    # monta data/rag/index.json
    python -m src.rag.search "Qual o DOI de 1twrmr?"
    python -m src.rag.eval_retrieval # Recall@k / MRR da recuperação
"""
