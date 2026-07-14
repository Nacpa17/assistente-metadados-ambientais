"""Roda o modelo treinado contra benchmarks/gold_qa.jsonl e gera relatório.

Uso:
    python -m src.eval.run_eval --config configs/exp01.yaml \
        --benchmark benchmarks/gold_qa.jsonl
    python -m src.eval.run_eval --config configs/exp01.yaml \
        --benchmark benchmarks/gold_qa.jsonl --baseline   # modelo base sem LoRA
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
    return model, tokenizer


@torch.inference_mode()
def generate_answer(model, tokenizer, system, question, gen_cfg):
    prompt = build_chat_prompt(system, history=[], new_user_message=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1536).to(model.device)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default="benchmarks/results")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    system = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    gen_cfg = cfg.get("generation", {})

    items = list(iter_jsonl(args.benchmark))
    if args.limit:
        items = items[: args.limit]
    log.info("Benchmark: %d perguntas", len(items))

    model, tokenizer = load_model_and_tokenizer(cfg, use_adapter=not args.baseline)

    preds, refs, kinds, latencies, records = [], [], [], [], []
    for i, item in enumerate(items, 1):
        question = item["prompt"]
        gold = item["completion"]
        kind = (item.get("meta") or {}).get("kind", "unknown")
        pred, dt = generate_answer(model, tokenizer, system, question, gen_cfg)
        preds.append(pred)
        refs.append(gold)
        kinds.append(kind.split("_")[0])
        latencies.append(dt)
        records.append({"i": i, "prompt": question, "gold": gold, "pred": pred,
                        "kind": kind, "latency_s": round(dt, 3)})
        if i % 10 == 0:
            log.info("Progresso %d/%d (média %.2fs/resposta)",
                     i, len(items), sum(latencies) / len(latencies))

    metrics = aggregate(preds, refs, kinds=kinds)
    metrics["latency_mean_s"] = round(sum(latencies) / max(1, len(latencies)), 3)
    metrics["latency_max_s"] = round(max(latencies) if latencies else 0, 3)
    metrics["model_mode"] = "baseline" if args.baseline else "finetuned"
    metrics["experiment_name"] = cfg.get("experiment_name", "exp")

    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = "baseline" if args.baseline else "finetuned"
    out_path = Path(args.out_dir) / \
        f"eval_{metrics['experiment_name']}_{suffix}_{ts}.json"
    save_json({"metrics": metrics, "predictions": records}, out_path)

    log.info("==== RESULTADO ====")
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
