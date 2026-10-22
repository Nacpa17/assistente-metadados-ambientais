"""Formatação de prompts — usada tanto no treino quanto na inferência.

Mantém um único lugar definindo a "linguagem" do modelo, evitando
divergência entre treino e API (problema comum de fine-tuning).
"""

from __future__ import annotations

from typing import Iterable


DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente técnico de instrumentação atmosférica especializado "
    "em metadados ambientais. Sempre responda em português, de forma objetiva "
    "e precisa. Se a informação não estiver disponível, diga isso claramente "
    "em vez de inventar."
)


def build_training_prompt(system: str, question: str) -> str:
    """Monta o prefixo de treino (tudo antes da resposta)."""
    system = system.strip()
    question = question.strip()
    return (
        f"<<SYS>>\n{system}\n<</SYS>>\n\n"
        f"Pergunta: {question}\n"
        f"Resposta:"
    )


def build_full_example(system: str, question: str, answer: str) -> tuple[str, str]:
    """Retorna (prefix, full_text) para mascarar loss apenas no prefixo."""
    prefix = build_training_prompt(system, question)
    full = f"{prefix} {answer.strip()}"
    return prefix, full


def build_chat_prompt(
    system: str,
    history: Iterable[dict],
    new_user_message: str | None = None,
) -> str:
    """Monta o prompt para inferência em modo chat com histórico."""
    parts = [f"<<SYS>>\n{system.strip()}\n<</SYS>>", ""]
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role == "user":
            parts.append(f"Pergunta: {content}")
        elif role == "assistant":
            parts.append(f"Resposta: {content}")
    if new_user_message is not None:
        parts.append(f"Pergunta: {new_user_message.strip()}")
    parts.append("Resposta:")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# RAG (fase 3)
# --------------------------------------------------------------------------- #
# A frase de recusa reaproveita a formulação usada no treino ("não está
# disponível nos metadados"), que o modelo já sabe produzir e que a métrica
# de recusa reconhece.
DEFAULT_RAG_INSTRUCTIONS = (
    "Responda usando as informações do CONTEXTO, que traz fichas recuperadas "
    "dos metadados ARM. Copie identificadores (DOIs, datas, códigos, "
    "coordenadas e nomes de datasets) exatamente como aparecem no contexto. "
    "Se a informação pedida não estiver no contexto, responda que essa "
    "informação não está disponível nos metadados."
)

NO_CONTEXT_TEXT = "(nenhuma ficha relevante foi encontrada nos metadados)"


def build_rag_prompt(
    system: str,
    history: Iterable[dict],
    context: str,
    new_user_message: str,
    instructions: str | None = DEFAULT_RAG_INSTRUCTIONS,
) -> str:
    """Prompt com contexto recuperado, no MESMO formato do treino.

    O contexto entra logo antes da pergunta que ele responde (e depois do
    histórico), para não confundir o modelo em conversas com vários turnos:

        <<SYS>> sistema + instruções de uso do contexto <</SYS>>
        (histórico: Pergunta/Resposta ...)
        Contexto (metadados recuperados):
        [1] Ficha de dataset ...
        Pergunta: ...
        Resposta:
    """
    sys_text = system.strip()
    if instructions and instructions.strip():
        sys_text += "\n\n" + instructions.strip()
    parts = [f"<<SYS>>\n{sys_text}\n<</SYS>>", ""]
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role == "user":
            parts.append(f"Pergunta: {content}")
        elif role == "assistant":
            parts.append(f"Resposta: {content}")
    parts.append("Contexto (metadados recuperados):\n"
                 + ((context or "").strip() or NO_CONTEXT_TEXT))
    parts.append("")
    parts.append(f"Pergunta: {new_user_message.strip()}")
    parts.append("Resposta:")
    return "\n".join(parts)
