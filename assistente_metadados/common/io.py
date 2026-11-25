"""Utilitários compartilhados pelos módulos do projeto."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def setup_logger(name: str = "ic", level: str = "INFO") -> logging.Logger:
    """Configura um logger simples com formatação consistente."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Linha {line_no} de {path} é JSON inválido: {e}")


def write_jsonl(items: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_json_files(json_dir: str | Path) -> list[Path]:
    """*.json do diretório do corpus, sem arquivos ocultos.

    Path.glob("*.json") também devolve ".state.json" (o estado do coletor),
    que não é um registro; subpastas (ex.: _orfaos/) ficam de fora.
    """
    return sorted(p for p in Path(json_dir).glob("*.json")
                  if not p.name.startswith(".") and p.is_file())
