"""Coletor robusto de metadados ARM (JSON-LD embutido em HTML).

- Coleta incremental via state file (não rebaixa URLs já coletadas com sucesso).
- Retries com backoff exponencial e rate limiting.
- Checksum SHA-256 para detectar mudanças no JSON-LD ao re-coletar.

Uso:
    python -m src.collect.arm_collector --config configs/collect.yaml
ou
    python -m src.collect.arm_collector --sitemap URL --out data/raw_json
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
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup

from src.common.io import ensure_dir, load_json, load_yaml, save_json, setup_logger

log = setup_logger("collect.arm")

JSONLD_TYPE = "application/ld+json"


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


def filename_for(url: str, dataset_obj: dict[str, Any]) -> str:
    name = dataset_obj.get("name") or ""
    base = slugify(name) if name else hashlib.sha1(url.encode()).hexdigest()[:12]
    return f"{base}.json"


def collect_one(session, url, cfg, state, out_dir):
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
    if prev and prev.get("sha256") == sha:
        return "skipped"
        
    fname = filename_for(url, dataset)
    save_json(dataset, out_dir / fname)
    state.urls[url] = {"sha256": sha, "filename": fname,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return "updated" if prev else "new"


def collect_source(cfg: ArmSourceConfig) -> dict[str, int]:
    out_dir = ensure_dir(cfg.output_dir)
    state_path = Path(cfg.state_file)
    state = State.load(state_path)

    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})

    log.info("Baixando sitemap: %s", cfg.sitemap_url)
    sm_resp = fetch_with_retry(session, cfg.sitemap_url,
                               cfg.request_timeout_seconds, cfg.max_retries)
    if sm_resp is None:
        log.error("Falha ao obter sitemap")
        return {"new": 0, "updated": 0, "skipped": 0, "failed": 0}

    urls = parse_sitemap_urls(sm_resp.content)
    
    # Aplica o filtro regex nas URLs encontradas no sitemap
    if cfg.url_include_regex:
        pat = re.compile(cfg.url_include_regex)
        urls = [u for u in urls if pat.search(u)]
        
    log.info("URLs candidatas após filtragem: %d", len(urls))

    counters = {"new": 0, "updated": 0, "skipped": 0, "failed": 0}
    for i, url in enumerate(urls, 1):
        try:
            result = collect_one(session, url, cfg, state, out_dir)
        except Exception as e:
            log.exception("Erro inesperado em %s: %s", url, e)
            result = "failed"
            
        counters[result] = counters.get(result, 0) + 1
        if i % 50 == 0:
            log.info("Progresso %d/%d %s", i, len(urls), counters)
            state.save(state_path)
        time.sleep(cfg.request_delay_seconds)

    state.save(state_path)
    log.info("Coleta encerrada: %s", counters)
    return counters


def main() -> None:
    p = argparse.ArgumentParser(description="Coletor de metadados ARM")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--sitemap", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--state", type=str, default=None)
    args = p.parse_args()

    if args.config:
        cfg_dict = load_yaml(args.config)
        for src in cfg_dict.get("sources", []):
            collect_source(ArmSourceConfig.from_dict(src))
    else:
        if not args.sitemap or not args.out:
            p.error("informe --config OU --sitemap e --out")
        collect_source(ArmSourceConfig(
            sitemap_url=args.sitemap, output_dir=args.out,
            state_file=args.state or f"{args.out}/.state.json"))


if __name__ == "__main__":
    main()