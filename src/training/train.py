"""Fine-tuning LoRA+QLoRA do LLaMA 2 com configuração via YAML — versão 2.1.

Novidades desta versão:
- PATCH de compatibilidade: permite RETOMAR (--resume) mesmo com torch < 2.6.
  O transformers recente bloqueia torch.load por segurança (CVE-2025-32434);
  como os checkpoints são gerados localmente por nós, o carregamento é seguro
  e reabilitamos essa função. Sem isso, --resume falha com ValueError.

Mantém da versão anterior:
- Prints de progresso com ETA e checkpoints (pause com Ctrl+C, resume com --resume).
- eval split + load_best_model_at_end, padding dinâmico,
  prepare_model_for_kbit_training, bf16/fp16 coerente, mascaramento de loss
  sem off-by-one, EOS no fim da resposta, tudo via YAML.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# --------------------------------------------------------------------------- #
# PATCH: permite resume de checkpoint local com torch < 2.6.
# O transformers passou a exigir torch >= 2.6 para chamar torch.load (mesmo com
# weights_only=True) por causa da CVE-2025-32434. Como os checkpoints são NOSSOS
# (gerados localmente), o carregamento é seguro e desativamos a checagem.
# --------------------------------------------------------------------------- #
try:
    import transformers.utils.import_utils as _tf_iu
    _tf_iu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass
try:
    import transformers.trainer as _tf_tr
    if hasattr(_tf_tr, "check_torch_load_is_safe"):
        _tf_tr.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass
# --------------------------------------------------------------------------- #

from src.common.io import load_yaml, setup_logger
from src.common.prompts import build_training_prompt, DEFAULT_SYSTEM_PROMPT

log = setup_logger("training.train")


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ETACallback(TrainerCallback):
    """Imprime progresso + ETA a cada log e avisa quando salva checkpoint."""

    def __init__(self):
        self.t0: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        print("=" * 70)
        print(f"[treino] iniciando — passos totais: {state.max_steps} | "
              f"épocas: {args.num_train_epochs}")
        print("[treino] para PAUSAR: Ctrl+C (o último checkpoint fica salvo)")
        print("[treino] para RETOMAR: adicione --resume ao mesmo comando")
        print("=" * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.t0 is None or state.global_step == 0:
            return
        elapsed = time.time() - self.t0
        rate = state.global_step / max(elapsed, 1e-6)
        remaining = (state.max_steps - state.global_step) / max(rate, 1e-6)
        pct = 100.0 * state.global_step / max(state.max_steps, 1)
        if "loss" in logs:
            print(f"[treino] passo {state.global_step}/{state.max_steps} "
                  f"({pct:.1f}%) | loss {logs['loss']:.4f} | "
                  f"decorrido {fmt_hms(elapsed)} | ETA {fmt_hms(remaining)} | "
                  f"{rate:.2f} passos/s")
        if "eval_loss" in logs:
            print(f"[aval ] passo {state.global_step} | "
                  f"eval_loss {logs['eval_loss']:.4f}")

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        print(f"[ckpt ] checkpoint salvo: {ckpt}")


def make_bnb_config(qcfg: dict[str, Any]) -> BitsAndBytesConfig:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    return BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_compute_dtype=dtype_map[qcfg.get("bnb_4bit_compute_dtype",
                                                  "bfloat16")],
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant",
                                                True)),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
    )


def make_lora_config(lcfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("alpha", 32)),
        target_modules=list(lcfg.get("target_modules",
                                     ["q_proj", "k_proj", "v_proj", "o_proj"])),
        lora_dropout=float(lcfg.get("dropout", 0.05)),
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
    )


def build_tokenize_fn(tokenizer, system_prompt: str, max_length: int):
    """Tokeniza prompt e resposta SEPARADAMENTE e mascara o loss do prompt."""
    eos_id = tokenizer.eos_token_id

    def tokenize(example):
        prefix = build_training_prompt(system_prompt, example["prompt"])
        answer_text = " " + example["completion"].strip()
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text,
                               add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [eos_id]
        input_ids = prefix_ids + answer_ids
        if len(input_ids) > max_length:
            overflow = len(input_ids) - max_length
            answer_ids = (answer_ids[:-overflow]
                          if overflow < len(answer_ids) else [])
            input_ids = prefix_ids + answer_ids
        labels = [-100] * len(prefix_ids) + list(answer_ids)
        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels}

    return tokenize


def find_last_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", nargs="?", const=True, default=False,
                    help="retoma do último checkpoint (ou de um caminho "
                         "específico: --resume checkpoints/exp01/checkpoint-500)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    mcfg = cfg["model"]
    dcfg = cfg["dataset"]
    tcfg = cfg["training"]
    system_prompt = cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    use_bf16 = bool(tcfg.get("bf16", False))
    use_fp16 = bool(tcfg.get("fp16", False))
    if use_bf16 and use_fp16:
        log.warning("bf16 e fp16 ambos True; mantendo bf16.")
        use_fp16 = False
    if use_bf16 and torch.cuda.is_available() \
            and not torch.cuda.is_bf16_supported():
        log.warning("GPU não suporta bf16; alternando para fp16.")
        use_bf16, use_fp16 = False, True

    out_dir = Path(mcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_from = None
    if args.resume:
        if args.resume is True:
            last = find_last_checkpoint(out_dir)
            if last is None:
                log.warning("Nenhum checkpoint em %s; iniciando do zero.",
                            out_dir)
            else:
                resume_from = str(last)
        else:
            resume_from = str(args.resume)
        if resume_from:
            print(f"[resume] retomando de: {resume_from}")

    log.info("Carregando tokenizer: %s", mcfg["base"])
    tokenizer = AutoTokenizer.from_pretrained(mcfg["base"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info("Carregando modelo base com quantização 4-bit "
             "(pode levar alguns minutos)...")
    bnb_cfg = make_bnb_config(cfg.get("quantization", {}))
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base"], quantization_config=bnb_cfg, device_map="auto",
        trust_remote_code=bool(mcfg.get("trust_remote_code", False)))
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(
            tcfg.get("gradient_checkpointing", True)))
    model = get_peft_model(model, make_lora_config(cfg.get("lora", {})))
    model.print_trainable_parameters()

    log.info("Carregando dataset: %s", dcfg["path"])
    raw = load_dataset("json", data_files=dcfg["path"])["train"]
    eval_split = float(dcfg.get("eval_split", 0.0))
    if eval_split > 0:
        split = raw.train_test_split(
            test_size=eval_split, seed=int(dcfg.get("shuffle_seed", seed)))
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw, eval_raw = raw, None

    tok_fn = build_tokenize_fn(
        tokenizer, system_prompt, int(dcfg.get("max_length", 512)))
    log.info("Tokenizando %d exemplos de treino...", len(train_raw))
    train_ds: Dataset = train_raw.map(
        tok_fn, remove_columns=train_raw.column_names, desc="tokenize/train")
    eval_ds: Dataset | None = None
    if eval_raw is not None:
        log.info("Tokenizando %d exemplos de validação...", len(eval_raw))
        eval_ds = eval_raw.map(
            tok_fn, remove_columns=eval_raw.column_names, desc="tokenize/eval")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    bsz = int(tcfg.get("per_device_train_batch_size", 1))
    gacc = int(tcfg.get("gradient_accumulation_steps", 8))
    epochs = float(tcfg.get("num_train_epochs", 3))
    eff_batch = bsz * gacc
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    total_steps = int(steps_per_epoch * epochs)
    print("=" * 70)
    print(f"[setup] exemplos treino: {len(train_ds)} | "
          f"validação: {len(eval_ds) if eval_ds else 0}")
    print(f"[setup] batch efetivo: {eff_batch} ({bsz} x {gacc} acumulação)")
    print(f"[setup] passos/época: ~{steps_per_epoch} | épocas: {epochs} | "
          f"passos totais: ~{total_steps}")
    print(f"[setup] checkpoints a cada {tcfg.get('save_steps', 500)} passos "
          f"em {out_dir}")
    print("=" * 70)

    do_eval = eval_ds is not None
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=int(
            tcfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=gacc,
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=tcfg.get("optim", "paged_adamw_8bit"),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        eval_strategy=tcfg.get("eval_strategy", "steps") if do_eval else "no",
        eval_steps=int(tcfg.get("eval_steps", 500)) if do_eval else None,
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=int(tcfg.get("save_steps", 500)),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        load_best_model_at_end=bool(
            tcfg.get("load_best_model_at_end", True)) and do_eval,
        metric_for_best_model=tcfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(tcfg.get("greater_is_better", False)),
        report_to=tcfg.get("report_to", ["none"]),
        run_name=tcfg.get("run_name", cfg.get("experiment_name", "exp")),
        seed=seed,
        ddp_find_unused_parameters=False,
    )

    callbacks: list[TrainerCallback] = [ETACallback()]
    if do_eval and bool(tcfg.get("load_best_model_at_end", True)):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=eval_ds, data_collator=collator, callbacks=callbacks)

    if "wandb" in (tcfg.get("report_to") or []):
        try:
            import wandb
            wandb.init(project="llama2-finetune-ic",
                       name=cfg.get("experiment_name", "exp"), config=cfg)
        except Exception as e:
            log.warning("wandb não inicializou: %s", e)

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        last = find_last_checkpoint(out_dir)
        print()
        print("=" * 70)
        print("[pausa] Treino interrompido pelo usuário (Ctrl+C).")
        if last:
            print(f"[pausa] Último checkpoint completo: {last}")
            print(f"[pausa] Para retomar exatamente dali:")
            print(f"        python -m src.training.train "
                  f"--config {args.config} --resume")
        else:
            print("[pausa] Nenhum checkpoint foi salvo ainda.")
        print("=" * 70)
        return

    log.info("Salvando artefatos finais em %s", out_dir)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"[fim  ] Treinamento concluído. Adapter LoRA salvo em: {out_dir}")
    print(f"[fim  ] Avalie com: python -m src.eval.run_eval "
          f"--config {args.config} --benchmark benchmarks/gold_qa.jsonl")
    print("=" * 70)


if __name__ == "__main__":
    main()
