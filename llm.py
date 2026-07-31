"""
llm.py — Cloud LLM orchestration supporting Groq and Gemini via Streamlit Secrets.
"""

import logging
import os
import streamlit as st
from typing import Iterator, Optional

from langchain_groq import ChatGroq
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
# You can choose your provider: "groq" or "gemini"
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq") # Change to "gemini" if you want to use Gemini by default

MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "3500"))
MAX_HISTORY_CHARS = int(os.environ.get("MAX_HISTORY_CHARS", "400"))
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))

# ----------------------------------------------------
# Prompt construction (Shared)
# ----------------------------------------------------
def _build_prompt(context: str, question: str, history: Optional[list[dict]] = None) -> tuple:
    context = context.strip()[:MAX_CONTEXT_CHARS]

    history_block = ""
    if history:
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
        last_assistant = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), None)
        if last_user and last_assistant:
            snippet = f"Previous question: {last_user}\nPrevious answer: {last_assistant}\n"
            history_block = snippet[:MAX_HISTORY_CHARS]

    system_prompt = (
        "You are a helpful assistant. Use the provided document context to answer the question when relevant. "
        "If the answer cannot be found in the context, use your own general knowledge and training to provide a helpful, accurate, and complete answer."
    )
    
    user_prompt = f"{history_block}Context:\n{context}\n\nQuestion: {question}"
    return system_prompt, user_prompt


# ----------------------------------------------------
# LLM Streaming & Blocking Implementation
# ----------------------------------------------------
def ask_llm_stream(context: str, question: str, history: Optional[list[dict]] = None) -> Iterator[str]:
    """Generator — yields text chunks using Groq or Gemini based on configuration/keys."""
    system_prompt, user_prompt = _build_prompt(context, question, history)
    
    # Try using Groq first if available in secrets
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if groq_api_key and AI_PROVIDER == "groq":
            llm = ChatGroq(
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=MAX_OUTPUT_TOKENS,
                api_key=groq_api_key
            )
            messages = [("system", system_prompt), ("human", user_prompt)]
            produced_any = False
            for chunk in llm.stream(messages):
                if chunk.content:
                    produced_any = True
                    yield chunk.content
            if produced_any:
                return
    except Exception:
        logger.warning("Groq failed or unavailable, trying fallback/Gemini...")

    # Fallback to Gemini if Groq isn't used or fails
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            yield "Error: Neither GROQ_API_KEY nor GEMINI_API_KEY is configured in Streamlit secrets!"
            return

        genai.configure(api_key=gemini_api_key)
        # Using Gemini 1.5 Flash for fast streaming responses
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )
        
        response = model.generate_content(user_prompt, stream=True)
        produced_any = False
        for chunk in response:
            if chunk.text:
                produced_any = True
                yield chunk.text
        if not produced_any:
            yield "I couldn't generate a response."
            
    except Exception:
        logger.exception("LLM streaming generation failed completely")
        yield "The assistant hit an error while generating a response. Please check your API keys in Streamlit secrets."


def ask_llm(context: str, question: str, history: Optional[list[dict]] = None) -> str:
    """Blocking call — returns the full answer as a string by consuming the stream."""
    return "".join(list(ask_llm_stream(context, question, history)))
