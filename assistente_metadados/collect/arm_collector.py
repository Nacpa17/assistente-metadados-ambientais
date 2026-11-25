"""Coletor robusto de metadados ARM (JSON-LD embutido em HTML).

- Coleta incremental via state file (não regrava registros que não mudaram).
- Retries com backoff exponencial e rate limiting.
- Checksum SHA-256 para detectar mudanças no JSON-LD ao re-coletar.
- Nome de arquivo ÚNICO por URL (título + hash da URL). Até a fase 2 o nome
  vinha só do título truncado em 80 caracteres: produtos com títulos longos
  e parecidos (ex.: "... 1-min average" x "... 30-min average") caíam no
  mesmo arquivo e um sobrescrevia o outro. Na próxima execução, as URLs que
  dividiam arquivo são recoletadas com nome único ("repaired") e o arquivo
  antigo, já sem dono, vai para <saida>/_orfaos/ (nada é apagado).
- Cada arquivo leva a procedência em "_fonte" (URL de origem, data e
  coletor); o SHA-256 é calculado só sobre o JSON-LD original.

Uso:
    python -m assistente_metadados.collect.arm_collector --config configs/collect.yaml
    python -m assistente_metadados.collect.arm_collector --config configs/collect.yaml --limite 20
    python -m assistente_metadados.collect.arm_collector --config configs/collect.yaml --out data/raw_json_v2
ou
    python -m assistente_metadados.collect.arm_collector --sitemap URL --out data/raw_json

Depois da coleta: python -m assistente_metadados.collect.audit (confere o corpus).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from assistente_metadados.common.io import (ensure_dir, list_json_files, load_json, load_yaml,
                           save_json, setup_logger)

log = setup_logger("collect.arm")

JSONLD_TYPE = "application/ld+json"
COLLECTOR_ID = "assistente_metadados.collect.arm_collector"
ORPHAN_DIR = "_orfaos"
PROVENANCE_KEY = "_fonte"


@dataclass
class ArmSourceConfig:
    name: str = "arm"
    sitemap_url: str = ""
    output_dir: str = "data/raw_json"
    state_file: str = "data/raw_json/.state.json"
    # Filtrar apenas páginas .html por padrão para evitar requisições inúteis nos links .xml do sitemap
    url_include_regex: str | None = r"\.html$"
    request_timeout_seconds: int = 30
    request_delay_seconds: float = 0.5
    max_retries: int = 3
    user_agent: str = "INPE-IC-Researcher/1.0"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArmSourceConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class State:
    urls: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls()
        data = load_json(path)
        return cls(urls=data.get("urls", {}))

    def save(self, path: Path) -> None:
        save_json({"urls": self.urls}, path)


def fetch_with_retry(session, url, timeout, max_retries):
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                log.warning("Status %s em %s (tentativa %d/%d); aguardando %.1fs",
                            resp.status_code, url, attempt, max_retries, delay)
                time.sleep(delay)
                delay *= 2
                continue
            log.warning("Status %s não retentável em %s", resp.status_code, url)
            return None
        except requests.RequestException as e:
            log.warning("Erro em %s (tentativa %d/%d): %s; aguardando %.1fs",
                        url, attempt, max_retries, e, delay)
            time.sleep(delay)
            delay *= 2
    log.error("Esgotadas as %d tentativas em %s", max_retries, url)
    return None


def parse_sitemap_urls(xml_bytes: bytes) -> list[str]:
    urls: list[str] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log.error("Sitemap malformado: %s", e)
        return urls
        
    # O ARM usa HTTPS no namespace do sitemap. 
    # Mapeamos ambas as variações (http e https) por segurança.
    namespaces = [
        {"sm": "https://www.sitemaps.org/schemas/sitemap/0.9"},
        {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    ]
    
    # Tenta extrair usando os namespaces definidos
    for ns in namespaces:
        locs = root.findall(".//sm:url/sm:loc", ns)
        if locs:
            for loc in locs:
                if loc.text:
                    urls.append(loc.text.strip())
            break # Se encontrou com um namespace, não precisa testar o outro
            
    # Caso seja um sitemap de índice (sitemaps dentro de sitemaps)
    if not urls:
        for ns in namespaces:
            locs = root.findall(".//sm:sitemap/sm:loc", ns)
            if locs:
                for loc in locs:
                    if loc.text:
                        urls.append(loc.text.strip())
                break

    return urls


def extract_jsonld_blocks(html: str) -> list[dict[str, Any]]:
    # Alterado para "html.parser" (nativo do Python) para evitar erros de dependência do lxml
    soup = BeautifulSoup(html, "html.parser")
    blocks: list[dict[str, Any]] = []
    
    for script in soup.find_all("script", attrs={"type": JSONLD_TYPE}):
        text = (script.string or script.get_text() or "").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # O ARM possui múltiplos scripts application/ld+json sequenciais no código fonte.
            # Essa etapa ignora blocos quebrados ou inválidos sem travar a execução.
            continue
        if isinstance(obj, list):
            blocks.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            blocks.append(obj)
    return blocks


def pick_dataset_block(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for b in blocks:
        t = b.get("@type") or b.get("type")
        if isinstance(t, list):
            if any(str(x).lower() == "dataset" for x in t):
                return b
        elif isinstance(t, str) and t.lower() == "dataset":
            return b
    # Caso não ache explicitamente um do tipo "Dataset", retorna o primeiro bloco válido
    return blocks[0] if blocks else None


def slugify(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:max_len] or "dataset"


def url_hash(url: str, n: int = 8) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:n]


def filename_for(url: str, dataset_obj: dict[str, Any]) -> str:
    """Nome legível (título) + hash da URL: duas URLs nunca dividem arquivo
    e a mesma URL sempre gera o mesmo nome."""
    name = dataset_obj.get("name") or ""
    base = slugify(name) if name else "dataset"
    return f"{base}__{url_hash(url)}.json"


def shared_filenames(state: State) -> dict[str, list[str]]:
    """Arquivos que o estado atribui a MAIS DE UMA URL (colisão de nomes da
    versão antiga do coletor): nome -> URLs. Só uma delas está no disco."""
    by_file: dict[str, list[str]] = {}
    for url, info in state.urls.items():
        if info.get("filename"):
            by_file.setdefault(info["filename"], []).append(url)
    return {f: sorted(u) for f, u in by_file.items() if len(u) > 1}


def with_provenance(dataset: dict[str, Any], url: str, ts: str) -> dict[str, Any]:
    out = dict(dataset)
    out[PROVENANCE_KEY] = {"url": url, "coletado_em": ts,
                           "coletor": COLLECTOR_ID}
    return out


def collect_one(session, url, cfg, state, out_dir, shared=frozenset()):
    resp = fetch_with_retry(session, url, cfg.request_timeout_seconds,
                            cfg.max_retries)
    if resp is None:
        return "failed"

    blocks = extract_jsonld_blocks(resp.text)
    dataset = pick_dataset_block(blocks)
    if not dataset:
        log.warning("Nenhum JSON-LD válido encontrado em %s", url)
        return "failed"

    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True)
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    prev = state.urls.get(url)
    prev_file = (prev or {}).get("filename")
    # o arquivo anterior só vale se existe e não é dividido com outra URL
    prev_ok = bool(prev_file) and prev_file not in shared \
        and (out_dir / prev_file).exists()
    if prev and prev.get("sha256") == sha and prev_ok:
        return "skipped"

    fname = prev_file if prev_ok else filename_for(url, dataset)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_json(with_provenance(dataset, url, ts), out_dir / fname)
    state.urls[url] = {"sha256": sha, "filename": fname, "ts": ts}
    if prev and not prev_ok:
        return "repaired"
    return "updated" if prev else "new"


def move_orphans(state: State, out_dir: Path, candidates) -> list[str]:
    """Arquivos antigos que nenhuma URL do estado referencia mais vão para
    <saida>/_orfaos/ (fora do corpus, mas preservados)."""
    referenced = {info.get("filename") for info in state.urls.values()}
    moved = []
    for fname in sorted(candidates):
        src = out_dir / fname
        if fname in referenced or not src.exists():
            continue
        dst = ensure_dir(out_dir / ORPHAN_DIR) / fname
        src.replace(dst)
        moved.append(fname)
    return moved


def new_counters() -> dict[str, int]:
    return {"new": 0, "updated": 0, "skipped": 0, "repaired": 0, "failed": 0}


def collect_source(cfg: ArmSourceConfig, session=None,
                   limit: int | None = None, force: bool = False) -> dict[str, int]:
    out_dir = ensure_dir(cfg.output_dir)
    state_path = Path(cfg.state_file)
    state = State.load(state_path)

    # Pasta com JSONs mas SEM estado (ex.: corpus copiado de outra máquina):
    # o coletor não sabe a que URL cada arquivo antigo corresponde e gravaria
    # tudo de novo com nomes novos -> corpus duplicado. Melhor coletar numa
    # pasta nova e comparar.
    existing = list_json_files(out_dir)
    if existing and not state.urls and not force:
        log.error("%s já tem %d JSONs, mas não há %s. Sem o estado, a coleta "
                  "gravaria cada registro de novo com nome novo e o corpus "
                  "ficaria DUPLICADO. Colete numa pasta nova (--out "
                  "data/raw_json_v2) e compare com a auditoria, ou use "
                  "--forcar se sabe o que está fazendo.", out_dir,
                  len(existing), state_path)
        return new_counters()

    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": cfg.user_agent})

    shared = shared_filenames(state)
    if shared:
        n_urls = sum(len(u) for u in shared.values())
        log.warning("Coleta anterior: %d arquivo(s) dividido(s) por %d URLs "
                    "(colisão de nomes; %d registro(s) sobrescrito(s)). Essas "
                    "URLs serão recoletadas com nome único.", len(shared),
                    n_urls, n_urls - len(shared))

    log.info("Baixando sitemap: %s", cfg.sitemap_url)
    sm_resp = fetch_with_retry(session, cfg.sitemap_url,
                               cfg.request_timeout_seconds, cfg.max_retries)
    if sm_resp is None:
        log.error("Falha ao obter sitemap")
        return new_counters()

    urls = parse_sitemap_urls(sm_resp.content)

    # Aplica o filtro regex nas URLs encontradas no sitemap
    if cfg.url_include_regex:
        pat = re.compile(cfg.url_include_regex)
        urls = [u for u in urls if pat.search(u)]

    log.info("URLs candidatas após filtragem: %d", len(urls))
    if limit:
        urls = urls[:limit]
        log.info("Limite de teste: processando só as %d primeiras", len(urls))

    counters = new_counters()
    for i, url in enumerate(urls, 1):
        try:
            result = collect_one(session, url, cfg, state, out_dir,
                                 shared=frozenset(shared))
        except Exception as e:
            log.exception("Erro inesperado em %s: %s", url, e)
            result = "failed"

        counters[result] = counters.get(result, 0) + 1
        if i % 50 == 0:
            log.info("Progresso %d/%d %s", i, len(urls), counters)
            state.save(state_path)
        time.sleep(cfg.request_delay_seconds)

    moved = move_orphans(state, out_dir, shared)
    if moved:
        log.info("%d arquivo(s) antigo(s) sem dono movido(s) para %s",
                 len(moved), out_dir / ORPHAN_DIR)
    still = shared_filenames(state)
    if still:
        log.warning("Ainda há %d arquivo(s) dividido(s) (URLs que falharam ou "
                    "ficaram fora do --limite). Rode a coleta de novo.",
                    len(still))
    state.save(state_path)
    log.info("Coleta encerrada: %s", counters)
    return counters


def main() -> None:
    p = argparse.ArgumentParser(description="Coletor de metadados ARM")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--sitemap", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--state", type=str, default=None)
    p.add_argument("--limite", type=int, default=None,
                   help="processa só as N primeiras URLs (teste rápido)")
    p.add_argument("--forcar", action="store_true",
                   help="coleta mesmo numa pasta com JSONs e sem .state.json")
    args = p.parse_args()

    if args.config:
        cfg_dict = load_yaml(args.config)
        for src in cfg_dict.get("sources", []):
            if args.out:   # --out troca a pasta de saída do config
                src = {**src, "output_dir": args.out,
                       "state_file": args.state or f"{args.out}/.state.json"}
            collect_source(ArmSourceConfig.from_dict(src), limit=args.limite,
                           force=args.forcar)
    else:
        if not args.sitemap or not args.out:
            p.error("informe --config OU --sitemap e --out")
        collect_source(ArmSourceConfig(
            sitemap_url=args.sitemap, output_dir=args.out,
            state_file=args.state or f"{args.out}/.state.json"),
            limit=args.limite, force=args.forcar)


if __name__ == "__main__":
    main()