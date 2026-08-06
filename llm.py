"""
llm.py — Cloud AI orchestration: Groq + Gemini (text) via LangChain/google-genai,
plus Hugging Face Inference API for image generation and video generation,
with rotation across multiple HF API tokens for resilience against
per-token rate limits.

NOTE ON SDK MIGRATION: this file previously used `google.generativeai`, which
Google deprecated on Nov 30, 2025 and does not support image/video generation
at all. It has been migrated to the current unified `google-genai` SDK
(`from google import genai`) for text chat, and to Hugging Face's Inference
API for image/video generation.

IMPORTANT — on multi-account key rotation:
Rotating across several HF tokens that belong to a *single* account (e.g.
separate tokens with different scopes, or a paid + free token) is fine.
Rotating across tokens from many separate accounts created specifically to
dodge one account's rate limit is against Hugging Face's Terms of Service
(most providers, HF included, prohibit multi-accounting to bypass limits),
and can get those accounts flagged/banned. This code will happily rotate
through however many tokens you give it — that policy risk is on the token
list you supply, not something this code can fix. For real production
scale, prefer HF PRO / Inference Endpoints (a paid, properly-provisioned
rate limit) over more free accounts.
"""

import io
import logging
import os
import random
import time
from typing import Iterator, Optional

import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_groq import ChatGroq
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.6-flash")

# Hugging Face Inference Providers — model IDs. As of mid-2025, HF's old
# legacy REST endpoint (api-inference.huggingface.co/models/<id>) mostly
# only serves small CPU models; GPU diffusion/video models are only
# reachable through Inference Providers routing (huggingface_hub's
# InferenceClient), which picks a real backend (fal-ai, Together,
# Replicate, etc.) to actually run the model. Override via env vars.
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
HF_VIDEO_MODEL = os.environ.get("HF_VIDEO_MODEL", "Wan-AI/Wan2.2-T2V-A14B")
# Provider to route through. "auto" lets HF pick the fastest available
# provider that serves the model; you can pin one (e.g. "fal-ai",
# "together", "replicate") if you want consistent behavior/pricing.
HF_IMAGE_PROVIDER = os.environ.get("HF_IMAGE_PROVIDER", "auto")
HF_VIDEO_PROVIDER = os.environ.get("HF_VIDEO_PROVIDER", "auto")

# How long (seconds) to let a token "cool down" after it gets rate-limited
# before we try it again.
HF_RATE_LIMIT_COOLDOWN = int(os.environ.get("HF_RATE_LIMIT_COOLDOWN", "60"))
# How long to wait for a cold model to finish loading on HF's side.
HF_MODEL_LOAD_TIMEOUT = int(os.environ.get("HF_MODEL_LOAD_TIMEOUT", "120"))

MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "40000"))
MAX_HISTORY_CHARS = int(os.environ.get("MAX_HISTORY_CHARS", "800"))
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))


class VideoGenerationUnavailable(Exception):
    """Raised when no working HF token/model is available for video."""


class ImageGenerationUnavailable(Exception):
    """Raised when no working HF token/model is available for images."""


# ----------------------------------------------------
# Hugging Face token rotation
# ----------------------------------------------------
# In-memory cooldown tracker: {token: unix_timestamp_until_which_it's_cold}
# This resets whenever the process restarts, which is fine — it's just
# meant to avoid hammering a token that *just* got a 429.
_hf_token_cooldowns: dict[str, float] = {}


def _get_hf_api_keys() -> list[str]:
    """Reads HF tokens from st.secrets or env var.

    Supports either:
      - HF_API_KEYS as a TOML array in secrets.toml:
            HF_API_KEYS = ["hf_abc...", "hf_def...", ...]
      - HF_API_KEYS as a comma-separated string (env var or secrets.toml):
            HF_API_KEYS = "hf_abc...,hf_def...,hf_ghi..."
      - A single HF_API_KEY as a fallback.
    """
    raw = None
    try:
        raw = st.secrets.get("HF_API_KEYS")
    except Exception:
        raw = None
    if raw is None:
        raw = os.environ.get("HF_API_KEYS")

    keys: list[str] = []
    if isinstance(raw, (list, tuple)):
        keys = [str(k).strip() for k in raw if str(k).strip()]
    elif isinstance(raw, str) and raw.strip():
        keys = [k.strip() for k in raw.split(",") if k.strip()]

    if not keys:
        single = None
        try:
            single = st.secrets.get("HF_API_KEY")
        except Exception:
            single = None
        single = single or os.environ.get("HF_API_KEY")
        if single:
            keys = [single.strip()]

    return keys


def _pick_hf_token(keys: list[str]) -> str:
    """Pick a token that isn't currently cooling down, preferring a random
    one so load is spread across all tokens rather than always hitting
    token[0] first. Falls back to the token whose cooldown ends soonest."""
    now = time.time()
    fresh = [k for k in keys if _hf_token_cooldowns.get(k, 0) <= now]
    if fresh:
        return random.choice(fresh)
    return min(keys, key=lambda k: _hf_token_cooldowns.get(k, 0))


def _mark_hf_token_limited(token: str) -> None:
    _hf_token_cooldowns[token] = time.time() + HF_RATE_LIMIT_COOLDOWN


def _hf_status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from an
    HfHubHTTPError (or any exception carrying a `.response`)."""
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) if resp is not None else None


def _hf_call_with_rotation(fn, model: str, provider: str):
    """Calls `fn(client)` (client = InferenceClient for one token), rotating
    through every token in HF_API_KEYS on rate limits (429) or auth errors
    (401/403) before giving up. On a "model loading" (503) response it
    retries the SAME token a few times instead of burning through others.
    Any other error is logged with its real detail and the next token is
    tried — nothing gets silently swallowed into a generic message.
    """
    keys = _get_hf_api_keys()
    if not keys:
        raise RuntimeError(
            "No Hugging Face API token configured. Set HF_API_KEYS (list or "
            "comma-separated) or HF_API_KEY in Streamlit secrets."
        )

    tried: set[str] = set()
    last_error: Optional[Exception] = None

    while len(tried) < len(keys):
        token = _pick_hf_token([k for k in keys if k not in tried] or keys)
        tried.add(token)
        client = InferenceClient(provider=provider, api_key=token)

        waited_for_load = 0
        while True:
            try:
                return fn(client)
            except HfHubHTTPError as e:
                status = _hf_status_code(e)
                detail = str(e)

                if status == 429:
                    logger.warning("HF token rate-limited, rotating to next token.")
                    _mark_hf_token_limited(token)
                    last_error = e
                    break  # try next token

                if status in (401, 403):
                    logger.warning("HF token invalid/unauthorized (%s): %s", status, detail)
                    last_error = e
                    break  # try next token

                if status == 503:
                    if waited_for_load >= HF_MODEL_LOAD_TIMEOUT:
                        last_error = TimeoutError("Model load timed out on Hugging Face.")
                        break
                    time.sleep(10)
                    waited_for_load += 10
                    continue  # retry same token — model is cold-starting

                # Any other status — log the REAL detail (model not found,
                # no provider available, bad payload, etc.) instead of
                # masking it, then try the next token.
                logger.warning("HF request failed (status=%s): %s", status, detail)
                last_error = e
                break
            except Exception as e:
                logger.warning("HF request failed on a token, trying next: %s", e)
                last_error = e
                break

    raise RuntimeError(
        f"All {len(keys)} Hugging Face token(s) failed. Last error: {last_error}"
    )


# ----------------------------------------------------
# Gemini client (text chat only — cached, one client per process)
# ----------------------------------------------------
_gemini_client: Optional["genai.Client"] = None


def _get_gemini_api_key() -> Optional[str]:
    return st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _get_gemini_client() -> "genai.Client":
    global _gemini_client
    if _gemini_client is None:
        api_key = _get_gemini_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured in Streamlit secrets.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ----------------------------------------------------
# Prompt construction
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

    if context:
        system_prompt = (
            "You are a helpful AI assistant. Use the provided document context to answer the question accurately."
        )
        user_prompt = f"{history_block}Context:\n{context}\n\nQuestion: {question}"
    else:
        system_prompt = (
            "You are a concise, direct, and factual AI assistant. Answer the user's questions clearly "
            "and accurately using your general knowledge without repeating yourself or stuttering."
        )
        user_prompt = f"{history_block}Question: {question}"

    return system_prompt, user_prompt


# ----------------------------------------------------
# Text chat: Groq first, Gemini fallback.
# ----------------------------------------------------
def ask_llm_stream(
    context: str,
    question: str,
    history: Optional[list[dict]] = None,
    images: Optional[list[tuple[bytes, str]]] = None,
) -> Iterator[str]:
    system_prompt, user_prompt = _build_prompt(context, question, history)

    if not images:
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if groq_api_key and AI_PROVIDER == "groq":
                llm = ChatGroq(
                    model=GROQ_MODEL,
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

    try:
        client = _get_gemini_client()
    except Exception as e:
        yield f"Error: {e}"
        return

    try:
        if images:
            parts = [types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images]
            parts.append(types.Part.from_text(text=user_prompt))
            contents = [types.Content(role="user", parts=parts)]
        else:
            contents = user_prompt

        produced_any = False
        for chunk in client.models.generate_content_stream(
            model=GEMINI_TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        ):
            if chunk.text:
                produced_any = True
                yield chunk.text
        if not produced_any:
            yield "I couldn't generate a response."
    except Exception:
        logger.exception("Gemini text generation failed")
        yield "The assistant hit an error while generating a response. Please check your API keys in Streamlit secrets."


def ask_llm(context: str, question: str, history: Optional[list[dict]] = None) -> str:
    return "".join(list(ask_llm_stream(context, question, history)))


# ----------------------------------------------------
# Image generation — Hugging Face Inference API, rotating across tokens.
# ----------------------------------------------------
def generate_image(prompt: str) -> bytes:
    """Generate an image from a text prompt via Hugging Face Inference
    Providers. Returns raw PNG bytes. Rotates across every token in
    HF_API_KEYS, skipping any token that's currently cooling down from a
    recent 429.

    Raises ImageGenerationUnavailable if every configured token is
    rate-limited/unauthorized, or no provider currently serves the model.
    """
    def _call(client: InferenceClient):
        image = client.text_to_image(prompt, model=HF_IMAGE_MODEL)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    try:
        return _hf_call_with_rotation(_call, model=HF_IMAGE_MODEL, provider=HF_IMAGE_PROVIDER)
    except RuntimeError as e:
        logger.exception("Image generation failed on every HF token")
        raise ImageGenerationUnavailable(
            "Image generation is temporarily unavailable — see the app logs "
            "for the real error from Hugging Face (invalid token, no "
            "provider serving this model, or genuine rate limiting). "
            "Please try again in a minute."
        ) from e


# ----------------------------------------------------
# Video generation — Hugging Face Inference API, rotating across tokens.
# ----------------------------------------------------
def generate_video(prompt: str) -> bytes:
    """Generate a video from a text prompt via Hugging Face Inference
    Providers. Returns raw MP4 bytes. Note: text-to-video models are
    slower and less consistently available across providers than image
    models — expect longer waits and occasional failures.

    Raises VideoGenerationUnavailable if every configured token is
    rate-limited/unauthorized, or no provider currently serves the model.
    """
    def _call(client: InferenceClient):
        return client.text_to_video(prompt, model=HF_VIDEO_MODEL)

    try:
        return _hf_call_with_rotation(_call, model=HF_VIDEO_MODEL, provider=HF_VIDEO_PROVIDER)
    except RuntimeError as e:
        logger.exception("Video generation failed on every HF token")
        raise VideoGenerationUnavailable(
            "Video generation is temporarily unavailable — see the app logs "
            "for the real error from Hugging Face (invalid token, no "
            "provider serving this model, or genuine rate limiting). "
            "Please try again shortly."
        ) from e


# ----------------------------------------------------
# Compatibility Wrapper
# ----------------------------------------------------
def get_llm():
    """Compatibility wrapper so app.py imports don't break."""
    return None
