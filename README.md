# Modelo de Linguagem Ajustado para Metadados Ambientais — IC PIBIC INPE (Fase 2)

Assistente de linguagem (LLM) especializado em metadados ambientais do repositório ARM. O objetivo é permitir que pesquisadores consultem, em português e linguagem natural, informações que hoje estão presas em arquivos JSON difíceis de manusear — por exemplo: *"qual instrumento mede umidade em Southern Great Plains?"*, *"o que é mean_doppler_velocity?"*, *"onde encontro dados de refletividade de radar?"*.

O projeto cobre: coleta de metadados, geração de um dataset de perguntas e respostas (incluindo um **glossário técnico** de termos de radar, lidar, radiação e aerossol, e **negativos** que ensinam o modelo a recusar o que não sabe), fine-tuning do LLaMA 2 7B com LoRA/QLoRA, avaliação por métricas automáticas, uma **API** e uma **interface web de chat** (no estilo do DataMap) para conversar com o modelo.

---

## Índice

1. [Instalação do zero (só tenho o VSCode)](#1-instalação-do-zero-só-tenho-o-vscode)
2. [Baixar o modelo LLaMA 2 (licença + login)](#2-baixar-o-modelo-llama-2-licença--login)
3. [Rodar o pipeline passo a passo](#3-rodar-o-pipeline-passo-a-passo)
4. [Acompanhar o treino (ETA) e pausar/retomar](#4-acompanhar-o-treino-eta-e-pausarretomar)
5. [Treinar no Google Colab (recomendado)](#5-treinar-no-google-colab-recomendado)
6. [Conversar com o modelo: interface web, API e CLI](#6-conversar-com-o-modelo-interface-web-api-e-cli)
7. [Estrutura do projeto](#7-estrutura-do-projeto)
8. [Solução de problemas](#8-solução-de-problemas)
9. [Nota sobre o método (fine-tuning × RAG)](#9-nota-sobre-o-método-fine-tuning--rag)
10. [Créditos](#10-créditos)

> **⚠️ As duas pegadinhas mais comuns**
> 1. **Rode sempre da raiz do projeto** (a pasta `projeto-ic`). Se aparecer `ModuleNotFoundError: No module named 'src'`, você não está na pasta certa.
> 2. **Local × Colab:** use `configs/exp01.yaml` para rodar/avaliar **no PC** (aponta para `checkpoints/exp01`), e `configs/exp01_colab.yaml` **só para treinar no Colab** (aponta para o Google Drive). Usar o config errado causa o erro `Can't find 'adapter_config.json'` — veja a [seção 8](#8-solução-de-problemas).

---

## 1. Instalação do zero (só tenho o VSCode)

Este guia presume que você tem **apenas o VSCode** instalado. Vamos instalar o resto.

### 1.1. Instalar o Python 3.11

1. Acesse <https://www.python.org/downloads/> e baixe o **Python 3.11.x** (evite o 3.12+ por ora, pela compatibilidade com bibliotecas de ML).
2. Execute o instalador. **MARQUE a caixa "Add Python to PATH"** na primeira tela (isso é essencial). Clique em "Install Now".
3. Para conferir, abra o **PowerShell** (menu Iniciar → digite "PowerShell") e rode:
   ```powershell
   python --version
   ```
   Deve aparecer `Python 3.11.x`.

### 1.2. Extensão Python no VSCode

1. Abra o VSCode.
2. Vá em Extensões (ícone de blocos na barra lateral, ou `Ctrl+Shift+X`).
3. Busque **"Python"** (da Microsoft) e instale. Instale também **"Pylance"** se ele não vier junto.

### 1.3. Abrir o projeto

1. Descompacte o projeto numa pasta de fácil acesso.
2. No VSCode: **File → Open Folder** e selecione a pasta `projeto-ic`.

### 1.4. Criar o ambiente virtual (venv)

Abra o terminal integrado do VSCode com **Ctrl + `** (a tecla de crase/acento grave, acima do Tab) e rode, **um comando por vez**:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Se aparecer um erro vermelho de *"execução de scripts foi desabilitada"* ao ativar, rode isto **uma única vez** e tente ativar de novo:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Quando o venv estiver ativo, o começo da linha do terminal mostra `(.venv)`.

### 1.5. Selecionar o interpretador no VSCode

Aperte `Ctrl+Shift+P`, digite **"Python: Select Interpreter"** e escolha o que tem `.venv` no caminho (termina em `.venv\Scripts\python.exe`). Isso faz o VSCode usar o ambiente certo.

### 1.6. Instalar as dependências

O **PyTorch** precisa ser instalado separadamente, escolhendo a versão da sua máquina.

**Se você tem GPU NVIDIA** (recomendado para o treino):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Se você NÃO tem GPU** (dá para gerar o dataset, servir a interface e testar tudo, menos treinar rápido):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Depois, instale o restante:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Essa instalação baixa vários pacotes e pode levar alguns minutos.

---

## 2. Baixar o modelo LLaMA 2 (licença + login)

O LLaMA 2 é um modelo *gated*: você precisa aceitar a licença da Meta e se autenticar. **Sem isso, o download falha com erro 401.** Faça uma vez:

1. Crie uma conta no Hugging Face: <https://huggingface.co/join>.
2. Aceite a licença do modelo nesta página (botão de aceite no topo): <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>. A aprovação costuma ser rápida.
3. Crie um token de acesso em <https://huggingface.co/settings/tokens> (tipo "Read").
4. No terminal do VSCode (com o venv ativo), faça login colando o token quando pedido:
   ```powershell
   huggingface-cli login
   ```

Pronto — a partir daqui os scripts conseguem baixar o modelo base automaticamente na primeira execução do treino.

---

## 3. Rodar o pipeline passo a passo

> **Regra de ouro:** rode todos os comandos a partir da raiz do projeto (a pasta `projeto-ic`). Os imports `from src...` dependem disso.

### 3.1. Colocar os metadados

Copie seus arquivos JSON-LD da fase 1 para a pasta `data/raw_json`:

```powershell
Copy-Item "C:\caminho\para\seus\json\*.json" data\raw_json\
```

(Ou rode o coletor, se quiser baixar do zero — veja o passo 3.6.)

### 3.2. Analytics (estatísticas do corpus)

```powershell
python -m src.analytics.compute --in data/raw_json --out data/analytics/stats.json
```

Gera `data/analytics/stats.json` com as variáveis e temas mais frequentes, número de datasets em coleta contínua, cobertura temporal, etc.

### 3.3. Glossário técnico

```powershell
python -m src.glossary.build --json-dir data/raw_json --out data/glossary/glossary_full.json
```

Combina o glossário **curado** (definições técnicas escritas à mão em `data/glossary/glossary.yaml` — radar, lidar, termodinâmica, radiação, aerossol, siglas de instrumentos ARM) com definições **automáticas** extraídas das descrições de variáveis dos próprios JSONs.

> Para editar/expandir as definições, abra `data/glossary/glossary.yaml` e acrescente ou ajuste entradas. É um arquivo de texto simples.

### 3.4. Gerar o dataset de treino

```powershell
python -m src.prompts.generator --config configs/dataset_v3.yaml
```

Gera `data/processed/dataset_v3.jsonl` (descritivo, corpus/recomendação, glossário, multi-hop).

### 3.5. Adicionar negativos (ensina o modelo a recusar)

```powershell
python -m src.prompts.add_negatives --dataset data/processed/dataset_v3.jsonl --json-dir data/raw_json --n 2500
```

Injeta ~2500 exemplos de perguntas **sem resposta** (campos que não existem nos metadados, assuntos fora do escopo) cuja resposta correta é recusar. Isso corrige a tendência do modelo de inventar respostas. Para conferir a mistura final do dataset:

```powershell
python -c "import json; from collections import Counter; c=Counter(); [c.update([json.loads(l)['meta']['kind'].split('_')[0]]) for l in open('data/processed/dataset_v3.jsonl', encoding='utf-8')]; print('Total:', sum(c.values())); print(dict(c))"
```

### 3.6. (Opcional) Coletar metadados novos do ARM

```powershell
python -m src.collect.arm_collector --config configs/collect.yaml
```

A coleta é incremental: `data/raw_json/.state.json` guarda o que já foi baixado e re-execuções só pegam o que mudou.

### 3.7. Fine-tuning

**Precisa de GPU** (consome ~8–12 GB de VRAM). No PC, use `configs/exp01.yaml`. Se a sua GPU for pequena, o treino é lento — nesse caso, prefira o **Colab** (veja a [seção 5](#5-treinar-no-google-colab-recomendado)).

```powershell
python -m src.training.train --config configs/exp01.yaml
```

### 3.8. Avaliação

```powershell
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl
```

Gera um relatório em `benchmarks/results/` com BLEU, ROUGE-L, F1, exact match, acurácia de recusa e latência. Para comparar com o modelo base (sem fine-tuning), adicione `--baseline`.

---

## 4. Acompanhar o treino (ETA) e pausar/retomar

### Progresso no terminal

Durante o treino, o script imprime um resumo inicial e, a cada poucos passos, uma linha de progresso com **tempo estimado para terminar (ETA)**:

```
[setup] exemplos treino: 29633 | validação: 605
[setup] batch efetivo: 8 (2 x 4 acumulação)
[setup] passos/época: ~3704 | épocas: 2 | passos totais: ~7408
[treino] passo 340/7408 (4.6%) | loss 1.4230 | decorrido 00:07:12 | ETA 02:34:09 | 0.79 passos/s
[aval ] passo 500 | eval_loss 0.8600
[ckpt ] checkpoint salvo: checkpoints/exp01/checkpoint-500
```

### Pausar

Aperte **Ctrl + C** no terminal a qualquer momento. O script informa qual foi o último checkpoint salvo e como retomar.

### Retomar

```powershell
python -m src.training.train --config configs/exp01.yaml --resume
```

Continua exatamente do último checkpoint (modelo, otimizador e passo são restaurados). Para retomar de um checkpoint específico:

```powershell
python -m src.training.train --config configs/exp01.yaml --resume checkpoints/exp01/checkpoint-500
```

> Os checkpoints são salvos a cada 500 passos (ajustável em `save_steps`, no config). O `train.py` já traz um patch que permite retomar mesmo com versões antigas do PyTorch.

---

## 5. Treinar no Google Colab (recomendado)

Numa GPU pequena o treino pode levar dezenas de horas. No Colab (GPU T4 gratuita) as épocas completam em **2–4 h**. O projeto já inclui o notebook `IC_treino_colab.ipynb`.

1. Suba a pasta `projeto-ic` **inteira** (com `data/raw_json` e os arquivos do projeto) para o seu Google Drive em `MyDrive/projeto-ic`.
2. Abra o <https://colab.research.google.com>, **File → Upload notebook**, e envie o `IC_treino_colab.ipynb`.
3. **Ambiente de execução → Alterar o tipo → GPU**.
4. Rode as células na ordem. Elas instalam tudo, geram o dataset (com glossário + negativos), treinam salvando no Drive, avaliam e deixam você conversar com o modelo.

**Detalhes importantes do Colab:**

- O `configs/exp01_colab.yaml` usa `fp16` (a T4 **não** tem bf16 nativo; forçar bf16 deixa o treino ~10× mais lento) e salva os checkpoints no Drive (`output_dir: /content/drive/MyDrive/projeto-ic/checkpoints/exp01`).
- Se o Colab cair no meio, rode as primeiras células de novo e use a célula **"7b. RETOMAR"** (os checkpoints ficam no Drive).
- Ao terminar, **baixe a pasta `checkpoints/exp01` do Drive para o seu PC** se quiser servir a interface localmente.

> **Voltou para o PC?** Ao rodar/avaliar localmente, use `configs/exp01.yaml` (caminho local `checkpoints/exp01`), **não** o `_colab`.

---

## 6. Conversar com o modelo: interface web, API e CLI

### 6.1. Interface web de chat (jeito recomendado)

A interface é um único arquivo `frontend/index.html`, no estilo visual do **DataMap** (branco, preto, acento verde-sálvia, tipografia Segoe UI, motivo de pontos/nós). A própria API serve a interface **e abre o navegador sozinha**.

Da raiz do projeto, com o venv ativo:

```powershell
# Windows (PowerShell) — LOCAL usa exp01.yaml:
$env:IC_CONFIG_PATH = "configs/exp01.yaml"
python -m src.api.server
```

```bash
# Linux/macOS:
IC_CONFIG_PATH=configs/exp01.yaml python -m src.api.server
```

O que acontece:
1. A API sobe em `http://localhost:8000`.
2. Quando o modelo termina de carregar (aparece `Modelo pronto.` no terminal), o navegador abre sozinho na interface.
3. A interface conversa com a API no mesmo endereço (sem CORS).

**Recursos da interface:**

- **Streaming** (texto sendo escrito) pelo endpoint `/chat/stream`; dá para desligar nos Ajustes e usar o `/chat` comum.
- **Formatação** nas respostas: **negrito**, `código`, listas e links (DOIs e URLs viram clicáveis).
- **Tempo de resposta e nº de tokens** embaixo de cada resposta.
- **Tema** claro/escuro/automático. Preferências salvas no navegador.
- **Modo demonstração**: se a API não estiver no ar, a interface funciona com respostas de exemplo (útil para ver o visual).
- Botão **Ajustes**: endereço da API, chave, temperatura, tamanho da resposta, tema.

**Variáveis de ambiente úteis:**

| Variável | Para quê |
|---|---|
| `IC_CONFIG_PATH` | qual config o servidor usa (obrigatória) — local: `configs/exp01.yaml` |
| `IC_PORT` | porta (default 8000) |
| `IC_NO_BROWSER=1` | **não** abrir o navegador ao iniciar |
| `IC_API_KEY` | exige o header `X-API-Key` igual (preencha o mesmo nos Ajustes) |
| `IC_CORS_ORIGINS` | origens permitidas (default `*`) |

### 6.2. API REST (uso programático / documentação interativa)

Se quiser só a API (sem abrir a interface), use o uvicorn direto:

```powershell
$env:IC_CONFIG_PATH = "configs/exp01.yaml"
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

- Interface interativa (Swagger): <http://localhost:8000/docs>
- Health check: <http://localhost:8000/healthz>
- Métricas: <http://localhost:8000/metrics>
- Exemplo de chamada:
  ```powershell
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"O que e reflectivity?\"}"
  ```

### 6.3. CLI (conversa direto no terminal)

```powershell
python -m src.api.chat_cli --config configs/exp01.yaml
```

Comandos no chat: `/reset` limpa o histórico, `/sair` encerra.

---

## 7. Estrutura do projeto

```
projeto-ic/
├── configs/
│   ├── collect.yaml           # config da coleta
│   ├── dataset_v3.yaml        # config do gerador de QA (+ glossário)
│   ├── exp01.yaml             # treino/serviço LOCAL (output_dir = checkpoints/exp01)
│   └── exp01_colab.yaml       # treino no COLAB (fp16, output_dir no Google Drive)
├── data/
│   ├── raw_json/              # JSONs ARM crus (você coloca aqui)
│   ├── processed/             # dataset_v3.jsonl gerado
│   ├── analytics/             # stats.json
│   └── glossary/
│       ├── glossary.yaml      # glossário CURADO (edite à vontade)
│       └── glossary_full.json # curado + auto (gerado)
├── frontend/
│   └── index.html             # interface web de chat (estilo DataMap)
├── checkpoints/
│   └── exp01/                 # adapter LoRA treinado (adapter_config.json, adapter_model.safetensors, ...)
├── src/
│   ├── collect/arm_collector.py
│   ├── analytics/compute.py
│   ├── glossary/build.py           # monta o glossário
│   ├── prompts/
│   │   ├── generator.py            # gera o dataset de QA
│   │   └── add_negatives.py        # injeta negativos (recusa)
│   ├── training/train.py           # fine-tuning com ETA e resume
│   ├── eval/{metrics,run_eval}.py
│   ├── api/{server,chat_cli}.py    # server.py serve a interface + abre o navegador
│   └── common/{io,prompts}.py      # utilitários (formato de prompt único)
├── benchmarks/
│   ├── gold_qa.jsonl          # benchmark anotado (inclui negativos)
│   └── results/               # relatórios de avaliação
├── scripts/
│   ├── run_pipeline.sh        # pipeline completo (Linux/macOS)
│   └── run_pipeline.ps1       # pipeline completo (Windows)
├── IC_treino_colab.ipynb      # notebook para treinar no Google Colab
├── requirements.txt
└── README.md                  # este arquivo
```

---

## 8. Solução de problemas

| Problema | Solução |
|---|---|
| `ValueError: Can't find 'adapter_config.json' at '/content/drive/...'` **ou** `HFValidationError: Repo id must be in the form ...` | O `output_dir` do config aponta para o **Google Drive** (é o `exp01_colab.yaml`). Rodando **localmente**, use `configs/exp01.yaml` (que aponta para `checkpoints/exp01`). Confira que a pasta `checkpoints/exp01` tem o `adapter_config.json` e o `adapter_model.safetensors` — se treinou no Colab, baixe-a do Drive. |
| `ModuleNotFoundError: No module named 'src'` | Você não está na raiz do projeto. Rode `cd` até a pasta `projeto-ic` antes de executar. |
| Erro **401 / gated repo** ao baixar o modelo | Você não aceitou a licença do LLaMA 2 ou não fez login. Veja a [seção 2](#2-baixar-o-modelo-llama-2-licença--login). |
| `Set-ExecutionPolicy` / não consigo ativar o venv | Rode `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` uma vez e ative de novo. |
| `CUDA out of memory` no treino | No config, mantenha `per_device_train_batch_size: 1` e aumente `gradient_accumulation_steps` (ex.: 16). Feche outros programas que usam a GPU. |
| `bf16 is not supported on your GPU` (ex.: Colab T4) | O `train.py` já cai para fp16 sozinho em GPUs sem bf16. No `exp01_colab.yaml` isso já está fixado (`fp16: true`, `bnb_4bit_compute_dtype: "float16"`). |
| Erro do `bitsandbytes` no Windows | Use uma versão recente (`pip install -U bitsandbytes`, ≥ 0.43 tem suporte a Windows). Se persistir, rode dentro do WSL2 (Ubuntu). |
| A API responde **503** | O modelo ainda está carregando; espere a mensagem `Modelo pronto.` no log. |
| A API responde **401** | A variável `IC_API_KEY` está definida; passe o header `X-API-Key` (ou preencha nos Ajustes da interface), ou remova a variável. |
| Interface com bolinha cinza "modo demonstração" | A interface não achou a API. Confira o terminal e o campo "Endereço da API" nos Ajustes. |
| O navegador não abriu sozinho | Abra manualmente <http://localhost:8000>. Verifique se `IC_NO_BROWSER` não está definido. |
| Erro de **CORS** no console do navegador | Prefira `python -m src.api.server` (a interface é servida no mesmo endereço, sem CORS). |
| O treino é muito lento sem GPU | Esperado. Sem GPU, use o pipeline até o passo 3.5 (gerar dataset) e treine no **Colab** (seção 5). |

---

## 9. Nota sobre o método (fine-tuning × RAG)

O fine-tuning ensina o modelo a **memorizar e recuperar com fluência** os fatos do corpus atual. Isso funciona muito bem para definições (glossário), recomendações e recusa, mas a **recordação exata de um fato único por identificador** (o DOI, a data ou a citação exata de *um* dataset específico entre centenas parecidos) é o teto natural do fine-tuning — o modelo aprende o *formato* (um DOI parece `10.5439/XXXXXXX`), mas nem sempre acerta os dígitos exatos.

Perguntas sobre datasets adicionados *depois* do treino também exigem re-treinar. Se, mais adiante, o objetivo for respostas sempre atualizadas e recordação exata sem re-treino, o caminho natural é acoplar uma camada de **RAG** (busca nos JSONs + geração por cima) — e os índices que o gerador já constrói (variável→datasets, facility→datasets, etc.) seriam reaproveitados quase diretamente. Para a fase 2 do PIBIC, porém, o fine-tuning com este dataset ampliado é o experimento correto a fazer primeiro.

---

## 10. Créditos

Projeto de Iniciação Científica PIBIC — INPE.
Bolsista: Eduardo Ribeiro do Amparo Rodrigues de Souza.
Orientador: Dr. Alan James Peixoto Calheiros (INPE). Coorientador: Prof. Dr. Pedro Luiz Pizzigatti Corrêa (USP).
