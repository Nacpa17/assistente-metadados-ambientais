#!/usr/bin/env bash
# Pipeline ponta-a-ponta (Linux/macOS) - fase 2 + RAG da fase 3.
# Rode a partir da raiz do projeto: bash scripts/run_pipeline.sh
# Os passos 1-6 nao exigem GPU; os passos 7-9 (treino/avaliacao) exigem.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " IC PIBIC - Pipeline (v3.1 com glossario + RAG)"
echo "============================================================"

echo
echo "[1/9] (opcional) Coleta de metadados ARM + auditoria do corpus..."
echo "      Pule a coleta se ja tem os JSONs em data/raw_json."
# python -m src.collect.arm_collector --config configs/collect.yaml
# (opcional, precisa de internet) catalogo dos servicos oficiais da ARM
# (DOD/DQR/sitios), usado automaticamente pelo indice RAG do passo 6:
# python -m src.collect.arm_extra --config configs/collect.yaml
python -m src.collect.audit --json-dir data/raw_json

echo
echo "[2/9] Computando analytics agregadas..."
python -m src.analytics.compute --in data/raw_json --out data/analytics/stats.json

echo
echo "[3/9] Construindo glossario (curado + auto)..."
python -m src.glossary.build --json-dir data/raw_json --out data/glossary/glossary_full.json

echo
echo "[4/9] Gerando dataset v3.1..."
python -m src.prompts.generator --config configs/dataset_v3.yaml

echo
echo "[5/9] Adicionando negativos (ensina o modelo a recusar)..."
python -m src.prompts.add_negatives --dataset data/processed/dataset_v3.jsonl --json-dir data/raw_json --n 2500

echo
echo "[6/9] RAG: indice de busca + avaliacao da recuperacao (sem GPU)..."
python -m src.rag.build_index --json-dir data/raw_json --glossary data/glossary/glossary_full.json --out data/rag/index.json
python -m src.rag.eval_retrieval --index data/rag/index.json --dataset data/processed/dataset_v3.jsonl

echo
echo "[7/9] Fine-tuning (exp01) -- exige GPU..."
python -m src.training.train --config configs/exp01.yaml

echo
echo "[8/9] Avaliando o fine-tuning contra o benchmark..."
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl

echo
echo "[9/9] Avaliando a arquitetura hibrida (fine-tuning + RAG)..."
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl --rag

echo
echo "Pipeline concluido. Resultados em benchmarks/results/."
