# Fontes de metadados da ARM — análise para a fase 3

Objetivo da renovação (item ii): **ampliar o corpus** com novos observatórios e
projetos, mantendo a ARM como fonte. Esta nota lista as fontes oficiais da ARM
que dá para extrair de forma parecida com o JSON-LD da fase 2, o que cada uma
acrescenta e o que já está implementado.

> Critério: só serviços **da própria ARM**. Os formatos foram conferidos no
> código-fonte do **ACT** (*Atmospheric data Community Toolkit*, biblioteca
> oficial da ARM, versão 2.3.4), que consulta esses mesmos serviços. Espelhos
> de terceiros não entram como fonte primária.

## Resumo

| # | Fonte (ARM) | O que acrescenta | Acesso | Valor para o projeto | Situação |
|---|---|---|---|---|---|
| 1 | **DOD** — *Data Object Design* (`pcm.arm.gov/pcm/api/dods/<produto>.<nível>`) | variáveis de cada produto e nível (`a1`, `b1`, `c1`…) com **nome longo, unidade**, `standard_name`, dimensões; versões do produto | público, JSON | **alto**: o JSON-LD não traz unidades; o DOD é a definição oficial do arquivo | **implementado** (`assistente_metadados.collect.arm_extra`) |
| 2 | **DQR** — *Data Quality Reports* (`dqr-web-service.arm.gov/dqr_full/...`) | problemas de qualidade conhecidos por datastream: categoria, período, variáveis afetadas, descrição, sugestão | público, JSON | **alto para o RAG**: responde "esse dado é confiável no período X?" | **implementado** (só no RAG) |
| 3 | **Serviço de citação** (`adc.arm.gov/citationservice/...`) | citação oficial de cada datastream, com DOI | público | médio: **confirma** o datastream e o DOI | **implementado** (opcional, `--citacoes`) |
| 4 | **Índice de metadados da Data Discovery** (`adc.arm.gov/elastic/metadata/_search`) | facilities de cada sítio com coordenadas; outros campos a explorar | público (usado pelo ACT) | médio: fichas de sítio; possivelmente mais campos | **implementado** (sítios) + `--explorar-site` |
| 5 | **Página de metadados por datastream** (`adc.arm.gov/metadata/html/<datastream>.html`) | metatags `citation_*`, período, mentores, medições | público, HTML | médio: alternativa quando o JSON-LD falha | proposto |
| 6 | **Páginas de observatórios e campanhas** (`arm.gov/capabilities/observatories/...`, notícias e páginas de campanha) | descrição de sítios e campanhas — inclusive **GoAmazon/MAO (Manacapuru-AM)** | público, HTML (texto livre) | **alto para o INPE** (contexto Brasil), mas exige curadoria | proposto: fichas curadas |
| 7 | **ADRS** e **ARM Live** (serviços de dados) | lista de arquivos e datas disponíveis por datastream; download | **exige cadastro** (usuário + chave de acesso) | alto para "quais dias existem", mas é dado, não metadado | pedir credencial se for necessário |
| 8 | Espelhos (`catalog.data.gov`, OSTI) | cópias de registros da ARM | público | baixo: derivados e desatualizados | **não usar** como fonte |

## O que muda no projeto

- **Unidades oficiais** (DOD) → glossário (`assistente_metadados.glossary.build --extra`) e fichas
  do RAG. Variáveis de controle de qualidade (`qc_*`) ficam de fora.
- **Níveis de processamento e datastreams confirmados** (DOD + DQR/citação) →
  fichas de dataset do RAG. Um datastream só é "confirmado" quando a ARM
  devolveu algo para ele (ou quando foi listado à mão): nomes montados a partir
  do corpus nunca são apresentados como certos.
- **DQRs** → só no RAG. É informação que muda com o tempo (novos relatórios,
  períodos corrigidos); treinar com ela congelaria um retrato que envelhece.
- **Sítios e coordenadas** → fichas de sítio do RAG.

## Como rodar

```powershell
python -m assistente_metadados.collect.audit                       # 1. o que o corpus tem hoje
python -m assistente_metadados.collect.arm_extra --dry-run         # 2. plano de consultas (sem internet)
python -m assistente_metadados.collect.arm_extra --max-classes 20  # 3. teste pequeno
python -m assistente_metadados.collect.arm_extra                   # 4. tudo (com cache)
python -m assistente_metadados.rag.build_index                     # 5. índice com o catálogo
python -m assistente_metadados.glossary.build --json-dir data/raw_json --extra auto   # 6. (novo FT)
```

Detalhes na seção 8 do README.

## Cuidados

- **Ritmo:** 0,5 s entre requisições, cache local e `user_agent` com e-mail
  (de `configs/collect.yaml`). Rodar de novo só consulta o que falta.
- **Primeira execução real:** os formatos foram conferidos no código do ACT e
  testados com um servidor falso nos mesmos formatos, mas o ambiente onde o
  código foi escrito não acessa os servidores da ARM. Comece com
  `--max-classes 20` e confira o resultado com `python -m assistente_metadados.rag.search`.
- **Coletor da fase 2:** nomeava arquivos só pelo título (80 caracteres), o
  que podia fazer um produto sobrescrever outro. Corrigido; a auditoria mostra
  se isso aconteceu no corpus atual e a recoleta recupera os registros.

## Brasil (interesse do INPE)

Na campanha **GoAmazon 2014/15** a ARM operou o sítio `MAO` no Amazonas, com
as facilities M1 (AMF1) e S1/S10 em Manacapuru, S2 em Iranduba, S3 (AAF) em
Manaus e F1 (avião G-1), todas encerradas em 2015 (página de locais do MAO,
abaixo). A auditoria lista os registros do Brasil que já estão no corpus;
produtos MAO que faltarem podem entrar pelo coletor (sitemap) ou pela lista
`--datastreams` do `arm_extra` (ex.: `maoarmbeatmM1.c1`). Como o sítio foi
encerrado em 2015, o período desses produtos é fechado (2014–2015), o que os
torna bons candidatos também para o treino.

## Referências

- ACT — Atmospheric data Community Toolkit: <https://github.com/ARM-DOE/ACT>
- PCM / DOD (documentação do ADI): <https://adi.readthedocs.io/en/latest/pcm_dod.html>
- Relatórios de qualidade (DQR): <https://arm.gov/data/data-quality-program/data-quality-reporting>
- DOIs dos datastreams: <https://arm.gov/working-with-arm/acknowledging-arm/doi-guidance-for-datastreams>
- Página de metadados por datastream (exemplo MAO): <https://adc.arm.gov/metadata/html/maoarmbeatmM1.c1.html>
- Locais do sítio MAO (GoAmazon): <https://arm.gov/capabilities/observatories/mao/locations>
- "A Decade of GoAmazon" (ARM): <https://arm.gov/news/features/post/94653>
- ADRS (serviço de pedido de dados): <https://adc.arm.gov/docs/adrsws.html>
- ARM Live: <https://adc.arm.gov/armlive/>
