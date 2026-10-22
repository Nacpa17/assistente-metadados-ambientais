"""Compatibilidade do treino com transformers 4.x e 5.x.

O Colab instala a versão mais recente do transformers (pip install -U), e a
série 5.x removeu argumentos que o projeto usava. Este módulo não importa
torch nem transformers, para poder ser testado sem GPU.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any


def training_kwargs_compat(args_cls: type, kwargs: dict[str, Any],
                           log: logging.Logger | None = None) -> dict[str, Any]:
    """Ajusta os argumentos de TrainingArguments à versão instalada.

    - warmup_ratio (removido no transformers 5): vira warmup_steps fracionário
      (no 5.x, warmup_steps < 1 é a fração dos passos totais — mesmo efeito);
    - argumentos que a versão instalada não conhece são descartados com aviso.
    """
    kw = dict(kwargs)
    names = {f.name for f in dataclasses.fields(args_cls)}
    if "warmup_ratio" in kw and "warmup_ratio" not in names:
        ratio = kw.pop("warmup_ratio")
        if ratio and "warmup_steps" in names and not kw.get("warmup_steps"):
            kw["warmup_steps"] = float(ratio)
    dropped = sorted(k for k in kw if k not in names)
    for k in dropped:
        kw.pop(k)
    if dropped and log:
        log.warning("Argumentos de treino não suportados nesta versão do "
                    "transformers (ignorados): %s", dropped)
    return kw
