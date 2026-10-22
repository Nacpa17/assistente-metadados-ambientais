"""TrainingArguments em transformers 4.x e 5.x (o 5.x removeu warmup_ratio).
Classes falsas com os mesmos campos: o teste não precisa de torch."""

import dataclasses
import logging

from src.training.compat import training_kwargs_compat


@dataclasses.dataclass
class ArgsV4:
    output_dir: str = ""
    warmup_ratio: float = 0.0
    warmup_steps: int = 0


@dataclasses.dataclass
class ArgsV5:
    output_dir: str = ""
    warmup_steps: float = 0


def test_v4_mantem_warmup_ratio():
    kw = training_kwargs_compat(ArgsV4, {"output_dir": "x", "warmup_ratio": 0.03})
    assert kw == {"output_dir": "x", "warmup_ratio": 0.03}


def test_v5_vira_warmup_steps_fracionario_e_avisa_o_resto(caplog):
    kw = training_kwargs_compat(
        ArgsV5, {"output_dir": "x", "warmup_ratio": 0.03, "arg_removido": 1},
        logging.getLogger("teste"))
    assert kw == {"output_dir": "x", "warmup_steps": 0.03}
    ArgsV5(**kw)                                   # constrói sem TypeError
    assert "arg_removido" in caplog.text


def test_v5_respeita_warmup_steps_explicito():
    kw = training_kwargs_compat(ArgsV5, {"warmup_ratio": 0.03, "warmup_steps": 100})
    assert kw == {"warmup_steps": 100}
