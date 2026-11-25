# Histórico de versões

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/), e as versões seguem o [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [2.0.0] - 2026-11-25

Fase 3 (renovação da bolsa) e reorganização do repositório.

### Adicionado

- Camada de RAG: fichas por conjunto de dados, termo, variável e observatório; busca por identificador exato, BM25 e ponte português → inglês; contexto com orçamento de tamanho; avaliação híbrida e Recall@k/MRR da recuperação.
- Clientes dos serviços oficiais da ARM (DOD, DQR, citação e índice) e auditoria do corpus.
- Métricas `contains_gold`, `false_refusal_rate` e por tipo de pergunta, e a reavaliação de resultados antigos sem rodar o modelo.
- Testes automáticos com mini-corpus sintético e integração contínua no GitHub Actions.

### Alterado

- O código passou do pacote `src` para o pacote `assistente_metadados`, e as dependências, dos `requirements*.txt` para o `pyproject.toml` (extras `treino`, `avaliacao`, `api`, `dev` e `completo`).
- O notebook foi para `notebooks/` e agora clona o repositório no Google Drive quando não acha a pasta do projeto.
- O passo a passo detalhado saiu do README para `docs/guia_completo.md`.
- O repositório foi renomeado para `assistente-metadados-ambientais` (os links antigos redirecionam).
- O treino ficou compatível com o transformers 5.

### Corrigido

- Frases de recusa que a avaliação não reconhecia.
- O coletor não duplica mais o corpus quando roda sem o arquivo de estado.

## [1.0.0] - 2026-07-14

Fase 2: coleta dos metadados, dataset com cerca de 30 mil pares pergunta-resposta e negativos, fine-tuning do LLaMA 2 7B com LoRA/QLoRA, avaliação por métricas automáticas, API REST e interface web de conversa.
