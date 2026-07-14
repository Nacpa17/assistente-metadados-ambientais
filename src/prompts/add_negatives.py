"""Adiciona negativos sintéticos (hard negatives) ao dataset de treino.

Motivo: no checkpoint parcial, refusal_accuracy = 0.0 — o modelo inventava
respostas técnicas para perguntas SEM resposta em vez de recusar. Isso
acontece porque o dataset tinha pouquíssimos exemplos de recusa (só campos
que por acaso faltavam num JSON). Aqui geramos muitos exemplos explícitos de
recusa para o modelo aprender a dizer "não consta" quando a informação não
existe.

Dois tipos:
  - campo impossível: pergunta sobre algo que NUNCA está nos metadados ARM
    (altitude de avião numa torre, custo, número de série, gerente, etc.)
  - fora de escopo: assunto alheio ao corpus (projetos INPE, cotação do
    dólar, previsão do tempo, etc.)

Rode DEPOIS do gerador:
    python -m src.prompts.generator --config configs/dataset_v3.yaml
    python -m src.prompts.add_negatives --dataset data/processed/dataset_v3.jsonl \
        --json-dir data/raw_json --n 2500
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.common.io import iter_jsonl, load_json, setup_logger, write_jsonl

log = setup_logger("prompts.add_negatives")

# Respostas de recusa variadas — o modelo deve aprender o CONCEITO de recusar,
# não decorar uma frase única.
REFUSALS = [
    "Essa informação não está disponível nos metadados deste dataset.",
    "Os metadados ARM consultados não contêm esse tipo de informação.",
    "Não há registro dessa informação nos metadados; consulte o provedor dos dados.",
    "Essa informação não consta nos metadados ARM analisados.",
    "Esse detalhe não está descrito nos metadados disponíveis para este conjunto de dados.",
    "Não encontro essa informação nos metadados; ela não faz parte do que a ARM disponibiliza para este dataset.",
]

# Perguntas sobre campos que NUNCA existem nos metadados JSON-LD da ARM.
# {name} é preenchido com nomes reais de datasets, gerando muita variedade.
IMPOSSIBLE_FIELD = [
    "Qual a altitude de voo do avião no dataset {name}?",
    "Quantas réplicas experimentais foram usadas em {name}?",
    "Qual o custo de aquisição dos dados de {name}?",
    "Qual o número de série do instrumento que gerou {name}?",
    "Qual a marca e o modelo do computador que processou {name}?",
    "Quantos pesquisadores trabalham no dataset {name} atualmente?",
    "Qual o orçamento anual do projeto que mantém {name}?",
    "Qual era a velocidade do vento no instante em que {name} foi publicado?",
    "Qual o peso do equipamento usado em {name}?",
    "Em que idioma os dados brutos de {name} foram anotados?",
    "Qual o horário de funcionamento do laboratório responsável por {name}?",
    "Qual a resolução da câmera fotográfica usada em {name}?",
    "Qual a cor do prédio onde {name} é armazenado?",
    "Quantas vezes {name} já foi baixado?",
    "Qual a nota de avaliação dos usuários para {name}?",
    "Qual o consumo de energia elétrica do instrumento de {name}?",
]

# Perguntas fora do escopo: não há dataset envolvido e o assunto é alheio
# ao corpus de metadados ARM.
OUT_OF_SCOPE = [
    "Qual a finalidade do projeto BIG do INPE?",
    "Como funciona o satélite CBERS-4A?",
    "Quais dados o INMET fornece sobre a cidade de São Paulo?",
    "Qual a previsão do tempo para amanhã em Brasília?",
    "Como faço o download de imagens do satélite GOES-16?",
    "Qual o telefone de contato da ARM?",
    "Quem é o diretor atual do INPE?",
    "Qual a cotação do dólar hoje?",
    "Como instalar o Python no Windows?",
    "Qual o melhor radar meteorológico para comprar?",
    "Quais os requisitos para uma bolsa de mestrado no INPE?",
    "Qual a diferença entre El Niño e La Niña?",
    "Quantos satélites o Brasil tem em órbita?",
    "Qual a altura da torre Eiffel?",
    "Como citar um artigo em formato ABNT?",
    "Qual a população da cidade de Lamont, Oklahoma?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset gerado (será reescrito com os negativos)")
    ap.add_argument("--json-dir", required=True,
                    help="pasta com os JSONs para pegar nomes reais")
    ap.add_argument("--n", type=int, default=2500,
                    help="quantos negativos sintéticos adicionar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="saída (default: sobrescreve --dataset)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    names = []
    for p in sorted(Path(args.json_dir).glob("*.json")):
        try:
            nm = (load_json(p).get("name") or "").strip()
            if nm:
                names.append(nm)
        except Exception:
            pass
    if not names:
        names = ["este dataset"]
    log.info("Nomes reais disponíveis: %d", len(names))

    existing = list(iter_jsonl(args.dataset))
    log.info("Dataset atual: %d exemplos", len(existing))
    seen = {(e["prompt"], e["completion"]) for e in existing}

    new: list[dict] = []
    attempts = 0
    while len(new) < args.n and attempts < args.n * 10:
        attempts += 1
        # 80% campo-impossível (espaço de combinações enorme), 20% fora-de-escopo
        if rng.random() < 0.8:
            q = rng.choice(IMPOSSIBLE_FIELD).format(name=rng.choice(names))
            kind = "negative_impossible_field"
        else:
            q = rng.choice(OUT_OF_SCOPE)
            kind = "negative_out_of_scope"
        a = rng.choice(REFUSALS)
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        new.append({"prompt": q, "completion": a,
                    "meta": {"kind": kind, "dataset": "_negativo"}})

    combined = existing + new
    rng.shuffle(combined)
    out = args.out or args.dataset
    total = write_jsonl(combined, out)
    log.info("Negativos sintéticos adicionados: %d", len(new))
    log.info("Salvo em %s (%d exemplos no total)", out, total)


if __name__ == "__main__":
    main()
