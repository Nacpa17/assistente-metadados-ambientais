"""Normalização e tokenização para a busca léxica (BM25).

O ponto delicado deste corpus são os IDENTIFICADORES: códigos de datastream
("1twrmr"), nomes de variáveis ("mean_doppler_velocity"), DOIs
("10.5439/1025256"). Um tokenizador comum quebraria tudo em pedaços e perderia
a correspondência exata, que é justamente o que o RAG precisa garantir.
Por isso cada identificador composto gera o token inteiro E as suas partes:

    "mean_doppler_velocity" -> mean_doppler_velocity, mean, doppler, velocity
    "10.5439/1025256"       -> 10.5439/1025256, 10, 5439, 1025256

Assim "1025256" ou "doppler" sozinhos ainda encontram o documento, mas o token
inteiro (raro, IDF alto) pesa mais quando a pergunta traz o identificador exato.
"""

from __future__ import annotations

import re
import unicodedata

# Sequências alfanuméricas ligadas por separadores típicos de identificadores.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/:\-]+[a-z0-9]+)*")
_TOKEN_RE_I = re.compile(_TOKEN_RE.pattern, re.IGNORECASE)
_SEP_RE = re.compile(r"[._/:\-]+")

# Palavras sem valor de busca (já sem acento, pois tudo passa por normalize).
STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele",
    "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em",
    "entre", "era", "essa", "essas", "esse", "esses", "esta", "estas",
    "este", "estes", "eu", "foi", "foram", "ha", "isso", "isto", "ja", "la",
    "lhe", "mais", "mas", "me", "mesmo", "meu", "minha", "muito", "na",
    "nao", "nas", "nem", "no", "nos", "nossa", "nosso", "num", "numa", "o",
    "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual",
    "quais", "quando", "que", "quem", "se", "seja", "sem", "ser", "seu",
    "sua", "suas", "seus", "so", "sao", "tambem", "te", "tem", "ter", "um",
    "uma", "umas", "uns", "voce", "onde", "quanto", "quantos", "quantas",
    "porque", "sobre", "existe", "existem", "possui", "posso", "pode",
    "preciso", "quero", "devo", "favor", "diga", "dizer", "aqui", "ai",
    "esta", "estao", "foi", "sido", "esteja", "deste", "desta", "neste",
    "nesta", "nesse", "nessa", "desse", "dessa", "cujo", "cuja",
    # termos genéricos do domínio que aparecem em quase toda pergunta
    "dataset", "datasets", "dado", "dados", "conjunto", "conjuntos",
    "produto", "produtos", "metadado", "metadados", "arm",
}
STOPWORDS_EN = {
    "the", "of", "and", "at", "in", "for", "to", "an", "is", "are", "by",
    "with", "from", "on", "this", "that", "these", "those", "be", "as",
    "or", "it", "its", "was", "were", "which", "into", "data",
}
STOPWORDS = STOPWORDS_PT | STOPWORDS_EN


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def normalize(s: str) -> str:
    """Minúsculas e sem acentos ("Período" -> "periodo")."""
    return strip_accents(s).lower()


def is_acronym(tok: str) -> bool:
    """"AOS", "SGP", "KAZR": siglas em maiúsculas (2+ letras)."""
    return len(tok) >= 2 and tok.isupper()


def tokenize(s: str, keep_stopwords: bool = False) -> list[str]:
    """Tokens para indexação/busca, preservando identificadores compostos.

    Siglas em MAIÚSCULAS nunca são descartadas como stopword: "AOS"
    (Aerosol Observing System) é sigla, "aos" é preposição.
    """
    out: list[str] = []
    for m in _TOKEN_RE_I.finditer(strip_accents(s)):
        raw = m.group(0)
        tok = raw.lower()
        if _SEP_RE.search(tok):
            out.append(tok)
            for part in _SEP_RE.split(tok):
                if part and (keep_stopwords or part not in STOPWORDS):
                    out.append(part)
        elif keep_stopwords or tok not in STOPWORDS or is_acronym(raw):
            out.append(tok)
    return out


def phrase(s: str, lower: bool = True) -> str:
    """Forma canônica para casar FRASES/identificadores inteiros.

    Mantém só caracteres alfanuméricos e separadores de identificador,
    com espaços simples e bordas marcadas — usada para checar se um
    identificador (ex.: nome completo de um dataset) aparece na pergunta.
    Com lower=False preserva maiúsculas (para siglas que colidem com
    palavras comuns, como AOS/aos).
    """
    text = normalize(s) if lower else strip_accents(s)
    toks = _TOKEN_RE_I.findall(text)
    return " " + " ".join(toks) + " " if toks else ""


def contains_phrase(haystack_phrase: str, needle: str) -> bool:
    """True se `needle` aparece inteiro (com bordas de palavra) no texto."""
    p = phrase(needle)
    return bool(p.strip()) and p in haystack_phrase


def doi_suffix(url: str | None) -> str | None:
    """Extrai '10.xxxx/yyyy' de uma URL de DOI (ou None)."""
    if not url:
        return None
    m = re.search(r"10\.\d{4,9}/[^\s\"'<>]+", url)
    return m.group(0).rstrip(".,;)") if m else None
