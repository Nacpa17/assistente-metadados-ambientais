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
