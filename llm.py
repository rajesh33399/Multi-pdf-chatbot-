"""
llm.py — Local LLM orchestration (TinyLlama via ctransformers).

- Lazily loads the model on first use and auto-downloads the GGUF weights
  from the Hugging Face Hub if they aren't already on disk.
- Serializes access to the shared model with a lock, since ctransformers
  is not safe for concurrent generate() calls on one instance, and Streamlit
  can serve multiple sessions from the same process.
- Fully generic strict-RAG prompting — no document-type-specific parsing.
"""

import logging
import os
import re
import threading
import time
from typing import Iterator, Optional

from huggingface_hub import hf_hub_download
# SWAPPED: Using CTransformers instead of llama_cpp
from langchain_community.llms import CTransformers

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Configuration (override via environment variables)
# ----------------------------------------------------
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

N_CTX = int(os.environ.get("LLM_N_CTX", "2048"))
N_THREADS = int(os.environ.get("LLM_N_THREADS", "6"))
N_BATCH = int(os.environ.get("LLM_N_BATCH", "32"))

MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "3500"))
MAX_HISTORY_CHARS = int(os.environ.get("MAX_HISTORY_CHARS", "400"))
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "128"))

NOT_FOUND_MSG = "Information not found in uploaded documents."

# ----------------------------------------------------
# Thread-safe lazy singleton
# ----------------------------------------------------
_llm_instance: Optional[CTransformers] = None
_load_lock = threading.Lock()
_inference_lock = threading.Lock()  # serializes calls across sessions


def _ensure_model_downloaded() -> str:
    """Download the GGUF from the Hub if it isn't already on disk."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Model not found locally, downloading %s/%s ...", MODEL_REPO_ID, MODEL_FILENAME)
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIR,
    )
    logger.info("Model downloaded to %s", downloaded_path)
    return downloaded_path


def get_llm() -> CTransformers:
    """Return the shared CTransformers instance, loading (and downloading, if needed) it
    on first use."""
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    with _load_lock:
        if _llm_instance is None:  # re-check inside the lock
            model_path = _ensure_model_downloaded()
            logger.info(
                "Loading TinyLlama from %s (n_ctx=%s, n_threads=%s)",
                model_path, N_CTX, N_THREADS,
            )

            _llm_instance = CTransformers(
                model=model_path,
                model_type="llama",
                config={
                    'max_new_tokens': MAX_OUTPUT_TOKENS,
                    'temperature': 0.1,
                    'top_p': 0.1,
                    'repetition_penalty': 1.1,
                    'context_length': N_CTX,
                    'threads': N_THREADS,
                    'batch_size': N_BATCH
                }
            )
            logger.info("TinyLlama loaded and ready.")
    return _llm_instance


# ----------------------------------------------------
# Prompt construction — optimized for small LLMs (TinyLlama)
# ----------------------------------------------------
def _build_prompt(context: str, question: str, history: Optional[list[dict]] = None) -> str:
    # FIXED: Replaced aggressive whitespace flattening (re.sub(r"\s+", " ")) 
    # which previously destroyed multi-line code blocks, YAML files, and lists.
    context = context.strip()[:MAX_CONTEXT_CHARS]

    history_block = ""
    if history:
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
        last_assistant = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), None)
        if last_user and last_assistant:
            snippet = f"Previous question: {last_user}\nPrevious answer: {last_assistant}\n"
            history_block = snippet[:MAX_HISTORY_CHARS]

    return f"""<|system|>
You are a helpful document assistant. Answer the question using only the provided context. If the answer is not in the context, reply with "{NOT_FOUND_MSG}".

<|user|>
{history_block}Context:
{context}

Question: {question}

<|assistant|>"""


def ask_llm(context: str, question: str, history: Optional[list[dict]] = None) -> str:
    """Blocking call — returns the full answer as a string."""
    if not context or not context.strip():
        return NOT_FOUND_MSG

    prompt = _build_prompt(context, question, history)

    t0 = time.time()
    try:
        llm = get_llm()
        with _inference_lock:
            answer = llm.invoke(prompt).strip()
        logger.info("LLM generation took %.1fs", time.time() - t0)
    except Exception:
        logger.exception("LLM generation failed")
        return "The assistant hit an error while generating a response. Please try again."

    if len(answer) < 3:
        return NOT_FOUND_MSG
    return answer


def ask_llm_stream(context: str, question: str, history: Optional[list[dict]] = None) -> Iterator[str]:
    """Generator — yields text chunks as they're produced."""
    if not context or not context.strip():
        yield NOT_FOUND_MSG
        return

    prompt = _build_prompt(context, question, history)

    t0 = time.time()
    logger.info("Starting streaming generation...")
    try:
        llm = get_llm()
        with _inference_lock:
            produced_any = False
            first_token_logged = False
            for token in llm.stream(prompt):
                if token:
                    if any(stop in token for stop in ["<|user|>", "<|system|>", "Question:"]):
                        break
                    if not first_token_logged:
                        logger.info("First token after %.1fs", time.time() - t0)
                        first_token_logged = True
                    produced_any = True
                    yield token
        logger.info("Streaming generation finished in %.1fs", time.time() - t0)
        if not produced_any:
            yield NOT_FOUND_MSG
    except Exception:
        logger.exception("LLM streaming generation failed")
        yield "The assistant hit an error while generating a response. Please try again."
