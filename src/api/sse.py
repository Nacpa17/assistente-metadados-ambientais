"""Formatação de eventos Server-Sent Events (SSE) para o /chat/stream.

Por que JSON: no protocolo SSE uma linha em branco ("\\n\\n") encerra o
evento. Antes, cada pedaço de texto ia cru em "data: <texto>\\n\\n"; quando o
modelo gerava uma quebra de linha (listas, parágrafos), o evento era cortado
no meio e o pedaço SEGUINTE da resposta sumia na interface. Serializando o
texto em JSON, a quebra vira "\\n" escapado e o evento nunca se parte.

Eventos emitidos pela API:
    event: sources  data: [{"n": 1, "title": ..., "doi": ...}, ...]   (RAG)
    data: "pedaço de texto"                                             (n vezes)
    event: done     data: "[END]"
    event: error    data: "mensagem"
"""

from __future__ import annotations

import json
from typing import Any


def sse_event(data: Any, event: str | None = None) -> str:
    """Um evento SSE com `data` em JSON (uma única linha, sempre)."""
    payload = json.dumps(data, ensure_ascii=False)
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n"


def parse_sse(stream: str) -> list[tuple[str | None, Any]]:
    """Inverso de sse_event (usado nos testes e por clientes Python)."""
    out: list[tuple[str | None, Any]] = []
    for frame in stream.split("\n\n"):
        if not frame.strip():
            continue
        event = None
        data = None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        out.append((event, data))
    return out
