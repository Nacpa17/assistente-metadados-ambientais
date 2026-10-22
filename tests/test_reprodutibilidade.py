"""Mesmos JSONs + mesma seed => mesmo dataset e mesmo índice, em QUALQUER
execução. A ordem de iteração de um set de strings muda a cada processo
(PYTHONHASHSEED); antes da correção, o gerador produzia datasets diferentes
de uma execução para outra mesmo com seed: 42."""

import json
import os
import subprocess
import sys

import yaml

from conftest import CURATED_GLOSSARY, ROOT


def _run(args, seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _pipeline(corpus_dir, out_dir, seed):
    cfg = yaml.safe_load((ROOT / "configs" / "dataset_v3.yaml")
                         .read_text(encoding="utf-8"))
    out_dir.mkdir()
    gl = out_dir / "glossary_full.json"
    _run(["src.glossary.build", "--json-dir", str(corpus_dir), "--curated",
          str(CURATED_GLOSSARY), "--out", str(gl), "--min-count", "1"], seed)
    cfg.update(input_json_dir=str(corpus_dir), glossary_path=str(gl),
               output_path=str(out_dir / "dataset.jsonl"))
    (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True),
                                      encoding="utf-8")
    _run(["src.prompts.generator", "--config", str(out_dir / "cfg.yaml")], seed)
    _run(["src.rag.build_index", "--json-dir", str(corpus_dir), "--glossary",
          str(gl), "--out", str(out_dir / "index.json"), "--extra", "none"], seed)
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    for volatile in ("built_at", "info"):
        index.pop(volatile, None)
    return ((out_dir / "dataset.jsonl").read_text(encoding="utf-8"),
            json.dumps(index, sort_keys=True))


def test_dataset_e_indice_nao_dependem_do_hash_seed(corpus_dir, tmp_path):
    a = _pipeline(corpus_dir, tmp_path / "a", seed=1)
    b = _pipeline(corpus_dir, tmp_path / "b", seed=2)
    assert a[0] == b[0], "o dataset gerado mudou entre execuções"
    assert a[1] == b[1], "o índice RAG mudou entre execuções"
