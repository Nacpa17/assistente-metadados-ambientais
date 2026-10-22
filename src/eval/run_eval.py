"""Roda o modelo treinado contra benchmarks/gold_qa.jsonl e gera relatório.

Uso:
    python -m src.eval.run_eval --config configs/exp01.yaml \
        --benchmark benchmarks/gold_qa.jsonl
    python -m src.eval.run_eval --config configs/exp01.yaml \
        --benchmark benchmarks/gold_qa.jsonl --baseline   # modelo base sem LoRA

Fase 3 — arquitetura híbrida (fine-tuning + RAG):
    python -m src.rag.build_index                          # uma vez
    python -m src.eval.run_eval --config configs/exp01.yaml \
        --benchmark benchmarks/gold_qa.jsonl --rag         # LoRA + RAG
    python -m src.eval.run_eval --config configs/exp01.yaml \
        --benchmark benchmarks/gold_qa.jsonl --baseline --rag   # base + RAG

As quatro combinações (base, base+RAG, LoRA, LoRA+RAG) formam a comparação
prevista no plano da renovação. No modo --rag cada predição registra as
fichas usadas ("sources") e o tempo de busca.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common.io import iter_jsonl, load_yaml, save_json, setup_logger
from src.common.prompts import build_chat_prompt, DEFAULT_SYSTEM_PROMPT
from src.eval.metrics import aggregate

log = setup_logger("eval.run")

# Limite do prompt sem RAG (igual ao da fase 2, para reprodutibilidade).
DEFAULT_MAX_INPUT_TOKENS = 1536


def load_model_and_tokenizer(cfg: dict[str, Any], use_adapter: bool):
    mcfg = cfg["model"]
    qcfg = cfg.get("quantization", {})
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_compute_dtype=dtype_map[qcfg.get("bnb_4bit_compute_dtype",
                                                  "bfloat16")],
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant",
                                                True)),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
    )
    log.info("Carregando modelo base: %s", mcfg["base"])
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["base"], quantization_config=bnb, device_map="auto",
        trust_remote_code=bool(mcfg.get("trust_remote_code", False)))
    if use_adapter:
        from peft import PeftModel
        log.info("Aplicando LoRA adapter de %s", mcfg["output_dir"])
        model = PeftModel.from_pretrained(base, mcfg["output_dir"])
        tokenizer_dir = mcfg["output_dir"]
    else:
        model = base
        tokenizer_dir = mcfg["base"]
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Se um prompt passar do limite, corta o INÍCIO (histórico/sistema) e
    # nunca a pergunta + "Resposta:" do fim — o corte padrão (à direita)
    # apagaria justamente a pergunta.
    tokenizer.truncation_side = "left"
    return model, tokenizer


@torch.inference_mode()
def generate_from_prompt(model, tokenizer, prompt: str, gen_cfg: dict,
                         max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_input_tokens).to(model.device)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 256)),
        do_sample=float(gen_cfg.get("temperature", 0.0)) > 0,
        temperature=float(gen_cfg.get("temperature", 0.2)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
        pad_token_id=tokenizer.eos_token_id)
    dt = time.time() - t0
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, dt


def generate_answer(model, tokenizer, system, question, gen_cfg):
    """Modo sem RAG (fase 2) — mantido com a mesma assinatura."""
    prompt = build_chat_prompt(system, history=[], new_user_message=question)
    return generate_from_prompt(model, tokenizer, prompt, gen_cfg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default="benchmarks/results")
    p.add_argument("--rag", action="store_true",
                   help="arquitetura híbrida: recupera fichas e põe no prompt")
    p.add_argument("--rag-index", default=None,
                   help="caminho do índice (padrão: rag.index_path do config "
                        "ou data/rag/index.json)")
    p.add_argument("--rag-k", type=int, default=None,
                   help="nº de fichas no contexto (padrão: rag.top_k ou 3)")
    p.add_argument("--max-input-tokens", type=int, default=None,
                   help="limite do prompt (padrão: 1536 sem RAG; "
                        "rag.max_input_tokens com RAG)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    system = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    gen_cfg = cfg.get("generation", {})

    items = list(iter_jsonl(args.benchmark))
    if args.limit:
        items = items[: args.limit]
    log.info("Benchmark: %d perguntas", len(items))

    rag = None
    if args.rag:
        # carrega o índice ANTES do modelo: se faltar, falha em segundos
        from src.rag.pipeline import RagPipeline
        rag = RagPipeline.from_config(
            cfg, index_path=args.rag_index, top_k=args.rag_k,
            max_input_tokens=args.max_input_tokens)
        log.info("RAG ligado: índice %s (%s documentos), top_k=%d, "
                 "limite do prompt=%d tokens", rag.settings.index_path,
                 rag.index.info.get("n_docs", len(rag.index.docs)),
                 rag.settings.top_k, rag.settings.max_input_tokens)

    model, tokenizer = load_model_and_tokenizer(cfg, use_adapter=not args.baseline)

    def count_tokens(s: str) -> int:
        return len(tokenizer(s)["input_ids"])

    max_input = (rag.settings.max_input_tokens if rag
                 else args.max_input_tokens or DEFAULT_MAX_INPUT_TOKENS)

    preds, refs, kinds, latencies, rlat, records = [], [], [], [], [], []
    for i, item in enumerate(items, 1):
        question = item["prompt"]
        gold = item["completion"]
        kind = (item.get("meta") or {}).get("kind", "unknown")
        rec: dict[str, Any] = {"i": i, "prompt": question, "gold": gold}
        if rag:
            res = rag.build_prompt(system, question, count_tokens=count_tokens)
            prompt = res.prompt
            rlat.append(res.retrieval_s)
            rec["sources"] = res.sources
            rec["retrieval_s"] = round(res.retrieval_s, 4)
        else:
            prompt = build_chat_prompt(system, history=[],
                                       new_user_message=question)
        pred, dt = generate_from_prompt(model, tokenizer, prompt, gen_cfg,
                                        max_input)
        total = dt + (rlat[-1] if rag else 0.0)
        preds.append(pred)
        refs.append(gold)
        kinds.append(kind.split("_")[0])
        latencies.append(total)
        rec.update({"pred": pred, "kind": kind, "latency_s": round(total, 3)})
        records.append(rec)
        if i % 10 == 0:
            log.info("Progresso %d/%d (média %.2fs/resposta)",
                     i, len(items), sum(latencies) / len(latencies))

    metrics = aggregate(preds, refs, kinds=kinds)
    metrics["latency_mean_s"] = round(sum(latencies) / max(1, len(latencies)), 3)
    metrics["latency_max_s"] = round(max(latencies) if latencies else 0, 3)
    mode = "baseline" if args.baseline else "finetuned"
    metrics["model_mode"] = mode + ("+rag" if rag else "")
    metrics["experiment_name"] = cfg.get("experiment_name", "exp")
    if rag:
        metrics["retrieval_latency_mean_s"] = round(
            sum(rlat) / max(1, len(rlat)), 4)
        s = rag.settings
        metrics["rag"] = {"index": s.index_path, "top_k": s.top_k,
                          "max_context_chars": s.max_context_chars,
                          "max_input_tokens": s.max_input_tokens,
                          "min_score": s.min_score,
                          "expand_pt_en": s.expand_pt_en,
                          "index_info": rag.index.info}

    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = mode + ("_rag" if rag else "")
    out_path = Path(args.out_dir) / \
        f"eval_{metrics['experiment_name']}_{suffix}_{ts}.json"
    save_json({"metrics": metrics, "predictions": records}, out_path)

    log.info("==== RESULTADO (%s) ====", metrics["model_mode"])
    for k, v in metrics.items():
        if not isinstance(v, dict):
            log.info("  %s = %s", k, v)
    if isinstance(metrics.get("token_f1_by_kind"), dict):
        log.info("  F1 por categoria:")
        for k, v in metrics["token_f1_by_kind"].items():
            log.info("    %-20s %.4f", k, v)
    log.info("Detalhes salvos em %s", out_path)


if __name__ == "__main__":
    main()
