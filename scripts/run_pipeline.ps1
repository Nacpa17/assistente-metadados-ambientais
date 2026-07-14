# Pipeline ponta-a-ponta da fase 2 (Windows / PowerShell).
# Rode a partir da raiz do projeto: .\scripts\run_pipeline.ps1
# Os passos 1-3 nao exigem GPU; o passo 4 (treino) exige.

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " IC PIBIC - Pipeline Fase 2 (v3.1 com glossario)"
Write-Host "============================================================"

Write-Host "`n[1/6] (opcional) Coleta de metadados ARM..."
Write-Host "      Pule se ja tem os JSONs em data\raw_json."
# python -m src.collect.arm_collector --config configs/collect.yaml

Write-Host "`n[2/6] Computando analytics agregadas..."
python -m src.analytics.compute --in data/raw_json --out data/analytics/stats.json

Write-Host "`n[3/6] Construindo glossario (curado + auto)..."
python -m src.glossary.build --json-dir data/raw_json --out data/glossary/glossary_full.json

Write-Host "`n[4/6] Gerando dataset v3.1 (descritivo + corpus + glossario + multi-hop)..."
python -m src.prompts.generator --config configs/dataset_v3.yaml

Write-Host "`n[5/6] Fine-tuning (exp01) -- exige GPU..."
python -m src.training.train --config configs/exp01.yaml

Write-Host "`n[6/6] Avaliando contra o benchmark..."
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl

Write-Host "`nPipeline concluido. Resultados em benchmarks\results\."
