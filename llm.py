"""
llm.py — Cloud LLM orchestration using Groq API (Lightning Fast & Free).
"""

import logging
import os
from typing import Iterator, Optional

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
# Using Groq's fast and free Llama 3.1 8B model
MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "3500"))
MAX_HISTORY_CHARS = int(os.environ.get("MAX_HISTORY_CHARS", "400"))
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))

NOT_FOUND_MSG = "Information not found in uploaded documents."

# ----------------------------------------------------
# LLM Singleton
# ----------------------------------------------------
def get_llm() -> ChatGroq:
    """Return the ChatGroq instance using the API key from environment variables."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set!")
    
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=MAX_OUTPUT_TOKENS,
        api_key=api_key
    )

# ----------------------------------------------------
# Prompt construction
# ----------------------------------------------------
def _build_prompt(context: str, question: str, history: Optional[list[dict]] = None) -> list:
    context = context.strip()[:MAX_CONTEXT_CHARS]

    history_block = ""
    if history:
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
        last_assistant = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), None)
        if last_user and last_assistant:
            snippet = f"Previous question: {last_user}\nPrevious answer: {last_assistant}\n"
            history_block = snippet[:MAX_HISTORY_CHARS]

    system_prompt = (
        "You are a helpful document assistant. Answer the question using only the provided context. "
        f'If the answer is not in the context, reply with "{NOT_FOUND_MSG}".'
    )
    
    user_prompt = f"{history_block}Context:\n{context}\n\nQuestion: {question}"

    return [
        ("system", system_prompt),
        ("human", user_prompt)
    ]


def ask_llm(context: str, question: str, history: Optional[list[dict]] = None) -> str:
    """Blocking call — returns the full answer as a string."""
    if not context or not context.strip():
        return NOT_FOUND_MSG

    messages = _build_prompt(context, question, history)

    try:
        llm = get_llm()
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception:
        logger.exception("LLM generation failed")
        return "The assistant hit an error while generating a response. Please check your GROQ_API_KEY."

    if len(answer) < 2:
        return NOT_FOUND_MSG
    return answer


def ask_llm_stream(context: str, question: str, history: Optional[list[dict]] = None) -> Iterator[str]:
    """Generator — yields text chunks as they're produced for token-by-token streaming."""
    if not context or not context.strip():
        yield NOT_FOUND_MSG
        return

    messages = _build_prompt(context, question, history)

    try:
        llm = get_llm()
        produced_any = False
        for chunk in llm.stream(messages):
            if chunk.content:
                produced_any = True
                yield chunk.content
        if not produced_any:
            yield NOT_FOUND_MSG
    except Exception:
        logger.exception("LLM streaming generation failed")
        yield "The assistant hit an error while generating a response. Please check your GROQ_API_KEY."
