"""Cliente de chat interativo para teste manual do modelo.

Uso:
    python -m src.api.chat_cli --config configs/exp01.yaml
    python -m src.api.chat_cli --config configs/exp01.yaml --deterministic
    python -m src.api.chat_cli --config configs/exp01.yaml --rag   # híbrido

Comandos no chat: /reset limpa o histórico, /sair encerra.
Com --rag, cada resposta mostra as fichas consultadas; /contexto mostra o
contexto completo enviado ao modelo na última pergunta.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from src.common.io import load_yaml, setup_logger
from src.common.prompts import build_chat_prompt, DEFAULT_SYSTEM_PROMPT

log = setup_logger("api.chat_cli")

MAX_HISTORY = 8


def load(cfg: dict):
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
    print("Carregando modelo base...")
    base = AutoModelForCausalLM.from_pretrained(
        mcfg["base"], quantization_config=bnb, device_map="auto",
        trust_remote_code=bool(mcfg.get("trust_remote_code", False)))
    print("Aplicando LoRA adapters...")
    model = PeftModel.from_pretrained(base, mcfg["output_dir"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(mcfg["output_dir"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # prompt longo demais: corta o início (histórico antigo), nunca a pergunta
    tokenizer.truncation_side = "left"
    return model, tokenizer


def print_sources(sources: list[dict]) -> None:
    if not sources:
        print("[fontes] nenhuma ficha relevante encontrada\n")
        return
    print("[fontes]")
    for s in sources:
        doi = f" — {s['doi']}" if s.get("doi") else ""
        print(f"  [{s['n']}] ({s['kind']}) {s['title']}{doi}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--rag", action="store_true",
                   help="busca fichas nos metadados e põe no prompt")
    p.add_argument("--rag-index", default=None)
    p.add_argument("--rag-k", type=int, default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.deterministic:
        set_seed(42)
    system = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    gen_cfg = cfg.get("generation", {})

    rag = None
    if args.rag:
        from src.rag.pipeline import RagPipeline
        rag = RagPipeline.from_config(cfg, index_path=args.rag_index,
                                      top_k=args.rag_k)
        print(f"RAG ligado: {rag.settings.index_path} "
              f"({rag.index.info.get('n_docs', len(rag.index.docs))} documentos)")

    model, tokenizer = load(cfg)
    history: list[dict] = []
    last_context = ""
    print("\nAssistente pronto. Digite '/sair' para encerrar; "
          "'/reset' para limpar o histórico"
          + ("; '/contexto' para ver o contexto da última pergunta" if rag
             else "") + ".\n")

    while True:
        try:
            user = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/sair", "sair", "exit", "quit"):
            print("Até logo.")
            break
        if user == "/reset":
            history = []
            print("[histórico limpo]\n")
            continue
        if user == "/contexto":
            print(f"\n{last_context or '(sem contexto ainda)'}\n")
            continue

        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        sources: list[dict] = []
        if rag is not None:
            res = rag.build_prompt(
                system, user, history=history,
                count_tokens=lambda s: len(tokenizer(s)["input_ids"]))
            prompt, sources, max_len = (res.prompt, res.sources,
                                        rag.settings.max_input_tokens)
            last_context = res.context
        else:
            prompt = build_chat_prompt(system, history, new_user_message=user)
            max_len = 1536
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_len).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.deterministic,
                temperature=0.0 if args.deterministic
                    else float(gen_cfg.get("temperature", 0.2)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                repetition_penalty=float(gen_cfg.get("repetition_penalty",
                                                    1.05)),
                pad_token_id=tokenizer.eos_token_id)
        new_ids = out[0, inputs["input_ids"].shape[-1]:]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        print(f"\nAssistente: {reply}\n")
        if rag is not None:
            print_sources(sources)
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
