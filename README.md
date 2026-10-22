# Modelo de Linguagem Ajustado para Metadados Ambientais — IC PIBIC INPE (Fases 2 e 3)

Assistente de linguagem (LLM) especializado em metadados ambientais do repositório ARM. O objetivo é permitir que pesquisadores consultem, em português e linguagem natural, informações que hoje estão presas em arquivos JSON difíceis de manusear — por exemplo: *"qual instrumento mede umidade em Southern Great Plains?"*, *"o que é mean_doppler_velocity?"*, *"onde encontro dados de refletividade de radar?"*.

O projeto cobre: coleta de metadados, geração de um dataset de perguntas e respostas (incluindo um **glossário técnico** de termos de radar, lidar, radiação e aerossol, e **negativos** que ensinam o modelo a recusar o que não sabe), fine-tuning do LLaMA 2 7B com LoRA/QLoRA, avaliação por métricas automáticas, uma **API** e uma **interface web de chat** (no estilo do DataMap) para conversar com o modelo.

**Fase 3 (renovação):** uma camada de **RAG** busca a ficha certa nos metadados e a coloca no prompt, para o modelo *ler* DOIs, datas e coordenadas em vez de tentar memorizá-los — a arquitetura híbrida fine-tuning + recuperação. Veja a [seção 7](#7-rag-consulta-aos-metadados-fase-3).

---

## Índice

1. [Instalação do zero (só tenho o VSCode)](#1-instalação-do-zero-só-tenho-o-vscode)
2. [Baixar o modelo LLaMA 2 (licença + login)](#2-baixar-o-modelo-llama-2-licença--login)
3. [Rodar o pipeline passo a passo](#3-rodar-o-pipeline-passo-a-passo)
4. [Acompanhar o treino (ETA) e pausar/retomar](#4-acompanhar-o-treino-eta-e-pausarretomar)
5. [Treinar no Google Colab (recomendado)](#5-treinar-no-google-colab-recomendado)
6. [Conversar com o modelo: interface web, API e CLI](#6-conversar-com-o-modelo-interface-web-api-e-cli)
7. [RAG: consulta aos metadados (fase 3)](#7-rag-consulta-aos-metadados-fase-3)
8. [Mais metadados da ARM: serviços oficiais e auditoria](#8-mais-metadados-da-arm-serviços-oficiais-e-auditoria)
9. [Estrutura do projeto](#9-estrutura-do-projeto)
10. [Solução de problemas](#10-solução-de-problemas)
11. [Nota sobre o método (fine-tuning × RAG)](#11-nota-sobre-o-método-fine-tuning--rag)
12. [Testes automáticos](#12-testes-automáticos)
13. [Créditos](#13-créditos)

> **⚠️ As duas pegadinhas mais comuns**
> 1. **Rode sempre da raiz do projeto** (a pasta `projeto-ic`). Se aparecer `ModuleNotFoundError: No module named 'src'`, você não está na pasta certa.
> 2. **Local × Colab:** use `configs/exp01.yaml` **no PC** e `configs/exp01_colab.yaml` **no Colab**. Os dois procuram o adapter em `checkpoints/exp01` **dentro da pasta do projeto** — no Colab, é a pasta do Drive em que a célula 3 do notebook entra. Se aparecer `Can't find 'adapter_config.json'`, veja a [seção 10](#10-solução-de-problemas).

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

> Com o catálogo de serviços oficiais da ARM (seção 8), `--extra auto` acrescenta as **unidades oficiais** dos DODs. Sem essa opção o glossário sai igual ao da fase 2.

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

Cada arquivo tem nome único (título + código da URL) e guarda a procedência em `_fonte` (URL de origem e data). Para um teste rápido: `--limite 20`. Depois da coleta, confira o corpus com `python -m src.collect.audit` (seção 8.1).

### 3.7. Fine-tuning

**Precisa de GPU** (consome ~8–12 GB de VRAM). No PC, use `configs/exp01.yaml`. Se a sua GPU for pequena, o treino é lento — nesse caso, prefira o **Colab** (veja a [seção 5](#5-treinar-no-google-colab-recomendado)).

```powershell
python -m src.training.train --config configs/exp01.yaml
```

### 3.8. Avaliação

```powershell
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl
```

Gera um relatório em `benchmarks/results/` com BLEU, ROUGE-L, F1, exact match, acurácia de recusa e latência. Para comparar com o modelo base (sem fine-tuning), adicione `--baseline`. Para a arquitetura híbrida (com RAG), adicione `--rag` — veja a [seção 7](#7-rag-consulta-aos-metadados-fase-3), que também explica as métricas novas (`contains_gold`, `false_refusal_rate`) e a correção da detecção de recusa.

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

1. Suba a pasta do projeto **inteira** (com `data/raw_json` e os arquivos do projeto) para o seu Google Drive — o notebook usa `MyDrive/projeto-ic-2`; se usar outro nome, ajuste `PROJETO` na célula 3.
2. Abra o <https://colab.research.google.com>, **File → Upload notebook**, e envie o `IC_treino_colab.ipynb`.
3. **Ambiente de execução → Alterar o tipo → GPU**.
4. Rode as células na ordem. Elas instalam tudo, geram o dataset (com glossário + negativos), treinam salvando no Drive, avaliam e deixam você conversar com o modelo.

**Detalhes importantes do Colab:**

- O `configs/exp01_colab.yaml` usa `bf16` (A100 e L4 suportam). Numa **T4**, o `train.py` troca para fp16 sozinho; troque também `bnb_4bit_compute_dtype` para `"float16"`, senão o treino fica bem mais lento.
- Os checkpoints vão para `checkpoints/exp01` **dentro da pasta do projeto no Drive** (caminho relativo, vale para qualquer nome de pasta) e sobrevivem se a sessão cair.
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

## 7. RAG: consulta aos metadados (fase 3)

A fase 2 mostrou que o fine-tuning acerta o **formato** das respostas, definições e recusas, mas tem um teto na **recordação exata** de fatos únicos (o DOI, a data ou as coordenadas de *um* dataset entre centenas parecidos). O RAG (*Retrieval-Augmented Generation*) resolve isso sem re-treinar: antes de responder, o sistema **busca a ficha certa** nos JSON-LD e a coloca no prompt; o modelo só precisa **ler** a informação.

> Tudo o que é busca roda em **Python puro** (sem GPU e sem bibliotecas novas): montar o índice, pesquisar e avaliar a recuperação levam segundos, mesmo no notebook. Só a etapa de gerar a resposta usa o LLaMA, como antes.

### 7.1. Montar o índice

Depois dos passos 3.1 (JSONs) e 3.3 (glossário):

```powershell
python -m src.rag.build_index
```

Gera `data/rag/index.json`. Reconstrua sempre que mudar os JSONs, o glossário ou o dicionário de expansão (7.6).

### 7.2. Testar a busca

```powershell
python -m src.rag.search "Qual o DOI de 1twrmr?"
python -m src.rag.search "O que significa a variável rh no dataset 1twrmr?" --contexto
python -m src.rag.search          # modo interativo: digite perguntas à vontade
```

A tabela mostra o tipo de ficha, a nota BM25 e se houve **casamento exato** de identificador. Com `--contexto` aparece exatamente o texto que iria para o modelo.

### 7.3. Avaliar a recuperação (Recall@k) — sem GPU

```powershell
python -m src.rag.eval_retrieval
```

Usa o próprio `data/processed/dataset_v3.jsonl` como gabarito (cada pergunta gerada sabe de qual dataset veio) e mede, por tipo de pergunta:

- **Recall@k**: em quantas perguntas a ficha certa aparece entre as *k* primeiras;
- **MRR**: média de 1/posição da ficha certa (1,0 = sempre em 1º lugar).

Também mostra quanto as perguntas **fora de escopo** "puxam" documentos (para calibrar `rag.min_score`). O relatório completo, com exemplos de erros, vai para `benchmarks/results/retrieval_*.json`. Para medir o efeito de cada componente: `--no-expand` (sem a ponte português→inglês) e `--no-focus` (sem o filtro de foco).

### 7.4. Avaliar o modelo com RAG (arquitetura híbrida)

```powershell
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl --rag
python -m src.eval.run_eval --config configs/exp01.yaml --benchmark benchmarks/gold_qa.jsonl --baseline --rag
```

Com os dois comandos da seção 3.8, isso fecha as quatro combinações do plano da renovação:

| | sem RAG | com RAG |
|---|---|---|
| **modelo base** | `--baseline` | `--baseline --rag` |
| **fine-tuning (LoRA)** | (padrão) | `--rag` |

No modo `--rag`, cada predição do relatório registra as fichas usadas (`sources`) e o tempo de busca.

### 7.5. Conversar com RAG (interface, API e CLI)

```powershell
# Windows (PowerShell)
$env:IC_CONFIG_PATH = "configs/exp01.yaml"
$env:IC_RAG = "1"
python -m src.api.server
```

```bash
# Linux/macOS
IC_CONFIG_PATH=configs/exp01.yaml IC_RAG=1 python -m src.api.server
```

A interface passa a mostrar **"Fontes consultadas"** embaixo de cada resposta (com o DOI clicável). Se o índice não existir, o servidor sobe **sem** RAG e avisa no log (e em `/healthz`). Também dá para ligar fixo com `rag.enabled: true` no config.

No terminal: `python -m src.api.chat_cli --config configs/exp01.yaml --rag` — o comando `/contexto` mostra o que foi enviado ao modelo na última pergunta.

### 7.6. Como funciona

- **Fichas** (`src/rag/documents.py`): uma por **dataset** (DOI, período, local, instrumento, variáveis, citação…), uma por termo do **glossário**, e fichas agregadas por **variável** (em quais datasets aparece), por **observatório** e para o **corpus** (estatísticas globais). As fichas usam as mesmas frases das respostas de treino (ex.: `1998-04-01 até o presente (coleta contínua)`), então o modelo encontra no contexto o formato que já sabe produzir.
- **Busca** (`src/rag/index.py`): primeiro o **casamento exato de identificadores** (nome completo, código `1twrmr`, sufixo do DOI, termo, variável, sigla do observatório); depois o **BM25** desempata e cobre perguntas sem identificador. Perguntas em português ganham equivalentes em inglês (*umidade → humidity*) pelo dicionário **`data/rag/expansoes_pt_en.yaml`** — edite à vontade, como o `glossary.yaml`.
- **Foco**: quando a pergunta cita um dataset pelo identificador, o contexto fica só com o que foi citado. Isso tira os "quase iguais" do prompt (ex.: `1twrmr` e `30twrmr` têm o mesmo título e DOIs diferentes).
- **Prompt** (`src/common/prompts.py`): o mesmo formato do treino (`<<SYS>> … Pergunta: … Resposta:`) com um bloco `Contexto (metadados recuperados)` antes da pergunta; se a informação não estiver no contexto, o modelo é instruído a dizer que não está disponível. As fichas menos relevantes saem até o prompt caber no limite de tokens (`rag.max_input_tokens`).
- **Ajustes** na seção `rag:` de `configs/exp01.yaml` (`top_k`, `max_context_chars`, `min_score`…).

### 7.7. Métricas novas e correção da detecção de recusa

- **Correção:** 2 das 6 frases de recusa usadas no treino (*"Esse detalhe não está descrito…"*, *"Não encontro essa informação…"*) e a resposta-ouro *"…fora do escopo…"* **não eram reconhecidas** como recusa, o que podia subestimar a `refusal_accuracy`. Para recalcular avaliações já feitas **sem rodar o modelo de novo**:
  ```powershell
  python -m src.eval.rescore "benchmarks/results/eval_*.json"
  ```
  (gera `*_rescored.json` ao lado do original e mostra o antes/depois).
- **`contains_gold`**: a resposta-ouro aparece dentro da resposta do modelo. Para fatos curtos, *"O DOI é X."* está certo mas zera o exact match; esta é a métrica de acerto usada em estudos de recordação de fatos (ex.: Kandpal et al., 2023).
- **`false_refusal_rate`**: em quantas perguntas **com** resposta o modelo recusou — o efeito colateral típico do RAG quando a busca falha.
- **`by_kind`**: as métricas acima separadas por categoria (descritivo, glossário, analítico, negativo).

---

## 8. Mais metadados da ARM: serviços oficiais e auditoria

O corpus da fase 2 vem das páginas da **ARM Data Discovery** (o JSON-LD de cada produto). A ARM publica outros serviços — os mesmos que a biblioteca oficial dela, o **ACT** (*Atmospheric data Community Toolkit*), consulta — com informações que o JSON-LD não traz. O módulo `src.collect.arm_extra` consulta esses serviços a partir do corpus que já temos (sem login) e junta tudo em `data/arm_extra/catalog.json`:

| Serviço oficial da ARM | O que acrescenta | Onde entra |
|---|---|---|
| **DOD** (*Data Object Design*) | variáveis de cada produto e nível (`a1`, `b1`, `c1`…) com **nome longo e unidade** oficiais | glossário (unidades) e fichas do RAG |
| **DQR** (*Data Quality Reports*) | problemas de qualidade conhecidos: categoria (*Incorrect*, *Suspect*…), período, variáveis afetadas, sugestão | só no RAG (muda com o tempo; não vai para o treino) |
| **Citação** | citação oficial de cada datastream, com DOI | confirma o datastream e o DOI |
| **Índice da Data Discovery** | facilities de cada sítio, com coordenadas | fichas de sítio do RAG |

A análise de todas as fontes avaliadas (o que vale a pena, o que exige cadastro, o que evitar) está em [`docs/fontes_metadados_arm.md`](docs/fontes_metadados_arm.md).

### 8.1. Auditar o corpus (comece por aqui)

```powershell
python -m src.collect.audit
```

Não acessa a internet. Mostra arquivos e registros válidos, duplicatas (por DOI, nome e conteúdo), a cobertura de cada campo (descrição, período, coordenadas, variáveis…), a distribuição por sítio e facility e os registros do **Brasil** (sítio MAO, campanha GoAmazon). O relatório completo vai para `data/analytics/auditoria_corpus.json`.

> **Importante:** até a fase 2, o coletor dava nome aos arquivos só pelo título do produto, cortado em 80 caracteres. Produtos com títulos longos e parecidos podiam cair no **mesmo arquivo**, e um sobrescrevia o outro. Se a auditoria mostrar **registros sobrescritos**, rode a coleta de novo (passo 3.6): o coletor agora usa nomes únicos, recupera esses registros e move o arquivo antigo para `data/raw_json/_orfaos/` (nada é apagado).

### 8.2. Coletar o catálogo

```powershell
python -m src.collect.arm_extra --dry-run          # mostra o plano, sem acessar a internet
python -m src.collect.arm_extra --max-classes 20   # teste pequeno: confira o resultado
python -m src.collect.arm_extra                    # todos os produtos do corpus
```

- As respostas ficam em cache (`data/arm_extra/cache`): dá para interromper e rodar de novo, que só busca o que falta. `--atualizar` ignora o cache (para pegar DQRs novos).
- Pausa entre requisições e identificação com e-mail (`request_delay_seconds` e `user_agent` de `configs/collect.yaml`), para não sobrecarregar os servidores da ARM.
- `--citacoes` busca também a citação oficial de cada datastream (mais lento).
- `--datastreams lista.txt` acrescenta datastreams pelo nome completo, um por linha (ex.: `maoarmbeatmM1.c1`) — útil para produtos do Brasil que ainda não estão no corpus.
- `--explorar-site mao` baixa uma amostra do índice da Data Discovery para um sítio e lista os campos existentes (para decidir o que mais extrair).
- Um datastream só aparece como **confirmado** quando a ARM devolveu algo para ele (DQR ou citação) ou quando você o listou: nomes montados a partir do corpus nunca são apresentados como certos.

### 8.3. Usar o catálogo

```powershell
python -m src.rag.build_index      # usa data/arm_extra/catalog.json se existir
python -m src.rag.search "Há problemas de qualidade conhecidos no 1twrmr?" --contexto
python -m src.rag.search "Quais as coordenadas das facilities do sítio SGP?"
```

- **RAG:** as fichas de dataset ganham os níveis de processamento, os datastreams confirmados (com DOI) e o resumo dos DQRs; as variáveis ganham **nome longo e unidade oficiais**; cada DQR e cada sítio viram fichas próprias. Uma pergunta sobre **qualidade** de um dataset citado traz os DQRs dele, os mais recentes primeiro. `--extra none` monta o índice sem o catálogo.
- **Glossário e novo fine-tuning:** `python -m src.glossary.build --json-dir data/raw_json --extra auto` acrescenta as unidades oficiais às entradas automáticas e cria entradas para variáveis que só existem nos DODs (sem as de controle de qualidade `qc_*`); as entradas curadas não mudam. **Sem `--extra`, o glossário sai igual ao da fase 2** — assim o baseline continua reprodutível. Use `--extra` ao montar o dataset do novo treino.

---

## 9. Estrutura do projeto

```
projeto-ic/
├── configs/
│   ├── collect.yaml           # config da coleta
│   ├── dataset_v3.yaml        # config do gerador de QA (+ glossário)
│   ├── exp01.yaml             # treino/serviço LOCAL (output_dir = checkpoints/exp01)
│   └── exp01_colab.yaml       # treino no COLAB (bf16, lotes maiores; checkpoints na pasta do projeto no Drive)
├── data/
│   ├── raw_json/              # JSONs ARM crus (você coloca aqui)
│   ├── processed/             # dataset_v3.jsonl gerado
│   ├── analytics/             # stats.json
│   ├── glossary/
│   │   ├── glossary.yaml      # glossário CURADO (edite à vontade)
│   │   └── glossary_full.json # curado + auto (gerado)
│   ├── rag/
│   │   ├── expansoes_pt_en.yaml  # ponte português -> inglês da busca (edite à vontade)
│   │   └── index.json            # índice RAG (gerado por src.rag.build_index)
│   └── arm_extra/                # catálogo dos serviços oficiais da ARM (gerado; seção 8)
│       ├── catalog.json
│       └── cache/
├── docs/
│   └── fontes_metadados_arm.md   # análise das fontes de metadados da ARM
├── frontend/
│   └── index.html             # interface web de chat (estilo DataMap)
├── checkpoints/
│   └── exp01/                 # adapter LoRA treinado (adapter_config.json, adapter_model.safetensors, ...)
├── src/
│   ├── collect/
│   │   ├── arm_collector.py        # coleta do JSON-LD (nomes únicos + procedência)
│   │   ├── arm_services.py         # clientes dos serviços oficiais (DOD, DQR, citação, índice)
│   │   ├── arm_extra.py            # CLI: monta data/arm_extra/catalog.json
│   │   └── audit.py                # CLI: auditoria do corpus
│   ├── analytics/compute.py
│   ├── glossary/build.py           # monta o glossário
│   ├── prompts/
│   │   ├── generator.py            # gera o dataset de QA
│   │   └── add_negatives.py        # injeta negativos (recusa)
│   ├── training/train.py           # fine-tuning com ETA e resume
│   ├── rag/                        # FASE 3 — recuperação (Python puro, sem GPU)
│   │   ├── documents.py            # fichas: dataset, glossário, variável, observatório, corpus
│   │   ├── extra.py                # fichas do catálogo ARM: DQR, sítio, unidades, níveis
│   │   ├── index.py                # busca: identificador exato + BM25 + PT->EN
│   │   ├── context.py              # monta o contexto do prompt (com orçamento de tamanho)
│   │   ├── pipeline.py             # busca + prompt (usado por avaliação, API e CLI)
│   │   ├── build_index.py          # CLI: monta data/rag/index.json
│   │   ├── search.py               # CLI: testa a busca
│   │   └── eval_retrieval.py       # CLI: Recall@k / MRR da recuperação
│   ├── eval/{metrics,run_eval,rescore}.py   # rescore = recalcula avaliações antigas
│   ├── api/{server,chat_cli,sse}.py         # server.py serve a interface + abre o navegador
│   └── common/                     # io, prompts (formato único), arm_names, arm_catalog
├── benchmarks/
│   ├── gold_qa.jsonl          # benchmark anotado (inclui negativos)
│   └── results/               # relatórios de avaliação
├── scripts/
│   ├── run_pipeline.sh        # pipeline completo (Linux/macOS)
│   └── run_pipeline.ps1       # pipeline completo (Windows)
├── tests/                     # testes automáticos (python -m pytest)
├── .github/workflows/testes.yml  # roda os testes no GitHub a cada push
├── IC_treino_colab.ipynb      # notebook para treinar no Google Colab
├── requirements.txt
├── requirements-dev.txt       # pytest (só para rodar os testes)
└── README.md                  # este arquivo
```

---

## 10. Solução de problemas

| Problema | Solução |
|---|---|
| `ValueError: Can't find 'adapter_config.json'` **ou** `HFValidationError: Repo id must be in the form ...` | O adapter não está em `checkpoints/exp01` da pasta onde você está rodando. Confira que está na raiz do projeto (no Colab: célula 3) e que `checkpoints/exp01` tem o `adapter_config.json` e o `adapter_model.safetensors` — se treinou no Colab e vai rodar no PC, baixe a pasta do Drive. Configs antigos tinham o caminho do Drive escrito por extenso (`/content/drive/MyDrive/projeto-ic/...`); use os desta versão. |
| `ModuleNotFoundError: No module named 'src'` | Você não está na raiz do projeto. Rode `cd` até a pasta `projeto-ic` antes de executar. |
| Erro **401 / gated repo** ao baixar o modelo | Você não aceitou a licença do LLaMA 2 ou não fez login. Veja a [seção 2](#2-baixar-o-modelo-llama-2-licença--login). |
| `Set-ExecutionPolicy` / não consigo ativar o venv | Rode `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` uma vez e ative de novo. |
| `CUDA out of memory` no treino | No config, mantenha `per_device_train_batch_size: 1` e aumente `gradient_accumulation_steps` (ex.: 16). Feche outros programas que usam a GPU. |
| `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'` | O transformers 5.x removeu `warmup_ratio`. O `train.py` desta versão já converte para o equivalente (`warmup_steps` fracionário); atualize o código. O notebook fixa as versões testadas (`transformers==5.20.2`, `peft==0.21.2`). |
| `bf16 is not supported on your GPU` (ex.: Colab T4) | O `train.py` já cai para fp16 sozinho em GPUs sem bf16. Numa T4, troque também `bnb_4bit_compute_dtype` para `"float16"` no `exp01_colab.yaml`. |
| Erro do `bitsandbytes` no Windows | Use uma versão recente (`pip install -U bitsandbytes`, ≥ 0.43 tem suporte a Windows). Se persistir, rode dentro do WSL2 (Ubuntu). |
| A API responde **503** | O modelo ainda está carregando; espere a mensagem `Modelo pronto.` no log. |
| A API responde **401** | A variável `IC_API_KEY` está definida; passe o header `X-API-Key` (ou preencha nos Ajustes da interface), ou remova a variável. |
| Interface com bolinha cinza "modo demonstração" | A interface não achou a API. Confira o terminal e o campo "Endereço da API" nos Ajustes. |
| O navegador não abriu sozinho | Abra manualmente <http://localhost:8000>. Verifique se `IC_NO_BROWSER` não está definido. |
| Erro de **CORS** no console do navegador | Prefira `python -m src.api.server` (a interface é servida no mesmo endereço, sem CORS). |
| O treino é muito lento sem GPU | Esperado. Sem GPU, use o pipeline até o passo 3.5 (gerar dataset) e treine no **Colab** (seção 5). |
| `Índice RAG não encontrado em data/rag/index.json` | Monte o índice: `python -m src.rag.build_index` (seção 7.1). |
| Com `IC_RAG=1` a interface não mostra "Fontes consultadas" | Veja o log ou <http://localhost:8000/healthz>: se `rag.enabled` estiver `false`, o campo `error` diz o motivo (geralmente o índice não foi montado). |
| A busca não acha um dataset que existe | Teste com `python -m src.rag.search "..."`. Citar o **código** (ex.: `1twrmr`) ou o nome completo ativa o casamento exato. Se os JSONs mudaram, reconstrua o índice. |
| Respostas com RAG ficaram mais lentas | O prompt fica maior (contexto). Diminua `rag.top_k` ou `rag.max_context_chars` no config. |
| Pergunta em português não acha dados em inglês | Acrescente a palavra em `data/rag/expansoes_pt_en.yaml` (ex.: `granizo: [hail]`) e reconstrua o índice. |
| A auditoria mostra **registros sobrescritos** | Colisão de nomes do coletor antigo. Rode `python -m src.collect.arm_collector --config configs/collect.yaml`: as URLs afetadas são recoletadas com nome único e o arquivo antigo vai para `data/raw_json/_orfaos/`. |
| `arm_extra` termina com muitas classes **sem DOD** | Quando o corpus não indica o nível, o `arm_extra` testa a lista de `--niveis` (padrão: `b1,c1,a1,c2,a0,s1`). Acrescente outros níveis a essa lista ou informe os datastreams completos em `--datastreams lista.txt`. |
| Erros 403/5xx ou "resposta não-JSON" no `arm_extra` | Rede bloqueada, serviço da ARM fora do ar ou com formato diferente. Rode de novo mais tarde (o que já veio fica no cache). Se persistir, abra no navegador a URL que aparece na mensagem de erro. |
| Quero o índice **sem** o catálogo da ARM | `python -m src.rag.build_index --extra none`. |

---

## 11. Nota sobre o método (fine-tuning × RAG)

O fine-tuning ensina o modelo a **memorizar e recuperar com fluência** os fatos do corpus atual. Isso funciona muito bem para definições (glossário), recomendações e recusa, mas a **recordação exata de um fato único por identificador** (o DOI, a data ou a citação exata de *um* dataset específico entre centenas parecidos) é o teto natural do fine-tuning — o modelo aprende o *formato* (um DOI parece `10.5439/XXXXXXX`), mas nem sempre acerta os dígitos exatos.

Perguntas sobre datasets adicionados *depois* do treino também exigem re-treinar. Por isso a fase 3 acopla a camada de **RAG** (seção 7): busca nos JSONs + geração por cima, reaproveitando quase diretamente os índices que o gerador já construía (variável→datasets, facility→datasets, etc.). A pergunta experimental agora é quanto a **arquitetura híbrida** (fine-tuning + RAG) melhora a recordação exata em relação ao fine-tuning puro — sem perder a qualidade em definições e recusas.

Um passo natural seguinte é **treinar já com o contexto no prompt** (o gerador sabe de qual ficha veio cada pergunta), para o modelo aprender a ler a ficha em vez de responder de memória.

---

## 12. Testes automáticos

Os testes cobrem a busca, o contexto, os prompts, as métricas, a avaliação da recuperação, a coleta nos serviços da ARM (com um **servidor falso** nos mesmos formatos), o glossário com unidades, o coletor e a auditoria, usando um **mini-corpus sintético** (não precisam dos JSONs reais, de internet nem de GPU):

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

No GitHub, o workflow `.github/workflows/testes.yml` roda os mesmos testes (Python 3.10 e 3.12) a cada push e pull request — dá para ver o resultado na aba **Actions**.

---

## 13. Créditos

Projeto de Iniciação Científica PIBIC — INPE.
Bolsista: Eduardo Ribeiro do Amparo Rodrigues de Souza.
Orientador: Dr. Alan James Peixoto Calheiros (INPE). Coorientador: Prof. Dr. Pedro Luiz Pizzigatti Corrêa (USP).
