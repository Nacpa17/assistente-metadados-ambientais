# Pipeline ponta-a-ponta (Windows / PowerShell) - fase 2 + RAG da fase 3.
# Rode a partir da raiz do projeto: .\scripts\run_pipeline.ps1
# Os passos 1-6 nao exigem GPU; os passos 7-9 (treino/avaliacao) exigem.

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " IC PIBIC - Pipeline (v3.1 com glossario + RAG)"
Write-Host "============================================================"

Write-Host "`n[1/9] (opcional) Coleta de metadados ARM + auditoria do corpus..."
Write-Host "      Pule a coleta se ja tem os JSONs em data\raw_json."
# python -m assistente_metadados.collect.arm_collector --config configs/collect.yaml
# (opcional, precisa de internet) catalogo dos servicos oficiais da ARM
# (DOD/DQR/sitios), usado automaticamente pelo indice RAG do passo 6:
# python -m assistente_metadados.collect.arm_extra --config configs/collect.yaml
python -m assistente_metadados.collect.audit --json-dir data/raw_json

Write-Host "`n[2/9] Computando analytics agregadas..."
python -m assistente_metadados.analytics.compute --in data/raw_json --out data/analytics/stats.json

Write-Host "`n[3/9] Construindo glossario (curado + auto)..."
python -m assistente_metadados.glossary.build --json-dir data/raw_json --out data/glossary/glossary_full.json

Write-Host "`n[4/9] Gerando dataset v3.1 (descritivo + corpus + glossario + multi-hop)..."
python -m assistente_metadados.prompts.generator --config configs/dataset_v3.yaml

Write-Host "`n[5/9] Adicionando negativos (ensina o modelo a recusar)..."
python -m assistente_metadados.prompts.add_negatives --dataset data/processed/dataset_v3.jsonl --json-dir data/raw_json --n 2500

Write-Host "`n[6/9] RAG: indice de busca + avaliacao da recuperacao (sem GPU)..."
python -m assistente_metadados.rag.build_index --json-dir data/raw_json --glossary data/glossary/glossary_full.json --out data/rag/index.json
python -m assistente_metadados.rag.eval_retrieval --index data/rag/index.json --dataset data/processed/dataset_v3.jsonl

Write-Host "`n[7/9] Fine-tuning (exp01) -- exige GPU..."
python -m assistente_metadados.training.train --config configs/exp01.yaml

Write-Host "`n[8/9] Avaliando o fine-tuning contra o benchmark..."
python -m assistente_metadados.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl

Write-Host "`n[9/9] Avaliando a arquitetura hibrida (fine-tuning + RAG)..."
python -m assistente_metadados.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl --rag

Write-Host "`nPipeline concluido. Resultados em benchmarks\results\."
