#!/usr/bin/env bash
# Pipeline ponta-a-ponta da fase 2 (Linux/macOS).
# Rode a partir da raiz do projeto: bash scripts/run_pipeline.sh
# Os passos 1-3 nao exigem GPU; o passo 5 (treino) exige.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " IC PIBIC - Pipeline Fase 2 (v3.1 com glossario)"
echo "============================================================"

echo
echo "[1/6] (opcional) Coleta de metadados ARM..."
echo "      Pule se ja tem os JSONs em data/raw_json."
# python -m src.collect.arm_collector --config configs/collect.yaml

echo
echo "[2/6] Computando analytics agregadas..."
python -m src.analytics.compute --in data/raw_json --out data/analytics/stats.json

echo
echo "[3/6] Construindo glossario (curado + auto)..."
python -m src.glossary.build --json-dir data/raw_json --out data/glossary/glossary_full.json

echo
echo "[4/6] Gerando dataset v3.1..."
python -m src.prompts.generator --config configs/dataset_v3.yaml

echo
echo "[5/6] Fine-tuning (exp01) -- exige GPU..."
python -m src.training.train --config configs/exp01.yaml

echo
echo "[6/6] Avaliando contra o benchmark..."
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl

echo
echo "Pipeline concluido. Resultados em benchmarks/results/."
