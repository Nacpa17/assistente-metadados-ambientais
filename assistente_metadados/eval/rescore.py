"""Recalcula as métricas de avaliações JÁ FEITAS, sem rodar o modelo de novo.

Os relatórios de assistente_metadados/eval/run_eval.py guardam todas as predições. Quando as
métricas mudam (ex.: a correção dos marcadores de recusa na fase 3), basta
re-pontuar os arquivos antigos — leva segundos e não precisa de GPU.

Uso:
    python -m assistente_metadados.eval.rescore benchmarks/results/eval_*.json
    python -m assistente_metadados.eval.rescore benchmarks/results/eval_X.json --no-write

Gera <arquivo>_rescored.json ao lado do original (o original não é alterado)
e mostra a comparação antes/depois das principais métricas.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from assistente_metadados.common.io import load_json, save_json, setup_logger
from assistente_metadados.eval.metrics import aggregate

log = setup_logger("eval.rescore")

KEYS = ["exact_match", "token_f1", "contains_gold", "refusal_accuracy",
        "false_refusal_rate", "bleu", "rouge_l"]
KEEP_FROM_ORIGINAL = ["latency_mean_s", "latency_max_s", "model_mode",
                      "experiment_name", "retrieval_latency_mean_s", "rag"]


def rescore_file(path: Path, write: bool = True) -> dict:
    data = load_json(path)
    preds_rec = data.get("predictions") or []
    if not preds_rec:
        raise ValueError(f"{path} não tem 'predictions'")
    preds = [r.get("pred", "") for r in preds_rec]
    refs = [r.get("gold", "") for r in preds_rec]
    kinds = [str(r.get("kind", "unknown")).split("_")[0] for r in preds_rec]
    old = data.get("metrics") or {}
    new = aggregate(preds, refs, kinds=kinds)
    for k in KEEP_FROM_ORIGINAL:
        if k in old:
            new[k] = old[k]
    new["rescored_from"] = path.name

    print(f"\n{path.name}  (n={len(preds)})")
    print(f"  {'métrica':<20}{'antes':>10}{'depois':>10}")
    for k in KEYS:
        a, b = old.get(k), new.get(k)
        if a is None and b is None:
            continue
        fa = "—" if a is None else f"{a:.4f}" if isinstance(a, float) else str(a)
        fb = "—" if b is None else f"{b:.4f}" if isinstance(b, float) else str(b)
        flag = "  <- mudou" if a is not None and b is not None and a != b else ""
        print(f"  {k:<20}{fa:>10}{fb:>10}{flag}")

    if write:
        out = path.with_name(path.stem + "_rescored.json")
        save_json({"metrics": new, "metrics_original": old,
                   "predictions": preds_rec}, out)
        print(f"  salvo em {out}")
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+",
                    help="arquivos eval_*.json (aceita curingas, também no Windows)")
    ap.add_argument("--no-write", action="store_true",
                    help="só mostra a comparação, sem salvar")
    a = ap.parse_args()

    paths: list[Path] = []
    for pattern in a.files:
        matches = glob.glob(pattern) or [pattern]
        paths.extend(Path(m) for m in matches)
    paths = [p for p in dict.fromkeys(paths)
             if not p.stem.endswith("_rescored")]
    if not paths:
        ap.error("nenhum arquivo encontrado")
    for p in paths:
        try:
            rescore_file(p, write=not a.no_write)
        except Exception as e:
            log.warning("Pulando %s: %s", p, e)


if __name__ == "__main__":
    main()
