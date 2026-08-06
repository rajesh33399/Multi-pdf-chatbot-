"""
llm.py — Cloud AI orchestration: Groq + Gemini (text) via LangChain/google-genai,
plus Gemini image generation (Nano Banana) and video generation (Veo).

NOTE ON SDK MIGRATION: this file previously used `google.generativeai`, which
Google deprecated on Nov 30, 2025 and does not support image/video generation
at all. It has been migrated to the current unified `google-genai` SDK
(`from google import genai`), which is required for Nano Banana / Veo access.
"""

import logging
import os
import time
from typing import Iterator, Optional

import streamlit as st
from langchain_groq import ChatGroq
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
# Current GA flash model as of mid-2026. Override via env var if Google ships
# a newer default before this code is updated again.
GEMINI_TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.6-flash")
# "Nano Banana" — Google's own migration guidance is to use this native
# multimodal model via generate_content, NOT the deprecated Imagen endpoint.
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
# Veo 3.1 preview — the 2.0/3.0 model lines were shut down June 30, 2026.
GEMINI_VIDEO_MODEL = os.environ.get("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview")

# NOTE: previously defaulted to 3500 chars (~700-900 tokens), which silently
# truncated any document longer than about a page — the tail of a resume
# (later projects, certifications, etc.) would just never reach the model.
# Both Groq's llama-3.1-8b-instant (128K token context) and Gemini flash
# models comfortably handle far more than this. 40000 chars (~8-10K tokens)
# covers multi-page resumes/documents with plenty of headroom to spare;
# raise further via the MAX_CONTEXT_CHARS env var if you're routinely
# feeding it longer documents.
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "40000"))
MAX_HISTORY_CHARS = int(os.environ.get("MAX_HISTORY_CHARS", "800"))
# Previously 512, which could cut off longer "list everything" style answers
# mid-sentence. 2048 gives enough room for a full multi-section answer.
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

VIDEO_POLL_SECONDS = int(os.environ.get("VIDEO_POLL_SECONDS", "10"))
VIDEO_MAX_WAIT_SECONDS = int(os.environ.get("VIDEO_MAX_WAIT_SECONDS", "300"))


class VideoGenerationUnavailable(Exception):
    """Raised when the configured API key/plan can't access video generation.
    app.py should catch this specifically and show a clear, honest message
    instead of a generic error — Veo access is gated by Google independently
    of anything in this code."""


class ImageGenerationUnavailable(Exception):
    """Raised when the configured API key/plan has zero quota for image
    generation (e.g. free-tier keys with limit: 0 on gemini-*-image models).
    This is a Google-side billing/plan restriction, not a bug — app.py
    should show a clear, honest message instead of the raw API error."""


def _is_quota_or_permission_error(msg: str) -> bool:
    return any(s in msg for s in (
        "PERMISSION_DENIED", "403", "not allowed", "not enabled",
        "quota", "RESOURCE_EXHAUSTED", "429",
    ))


# ----------------------------------------------------
# Gemini client (cached — one client per process, not per call)
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
# Text chat: Groq first, Gemini fallback. If images are attached, Groq is
# skipped entirely — ChatGroq's model here is text-only, so a turn with an
# image has to go straight to Gemini or the image would be silently dropped.
# ----------------------------------------------------
def ask_llm_stream(
    context: str,
    question: str,
    history: Optional[list[dict]] = None,
    images: Optional[list[tuple[bytes, str]]] = None,
) -> Iterator[str]:
    """Generator — yields text chunks using Groq or Gemini based on configuration/keys.

    `images` is an optional list of (raw_bytes, mime_type) tuples, e.g. from
    an uploaded PNG/JPEG. When present, this always routes to Gemini, since
    Groq's chat model here has no vision capability.
    """
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
    """Blocking call — returns the full answer as a string by consuming the stream."""
    return "".join(list(ask_llm_stream(context, question, history)))


# ----------------------------------------------------
# Image generation (Gemini "Nano Banana") — always via Gemini,
# regardless of AI_PROVIDER, since Groq has no image generation.
# ----------------------------------------------------
def generate_image(prompt: str) -> bytes:
    """Generate an image from a text prompt. Returns raw image bytes (PNG/JPEG).

    Raises ImageGenerationUnavailable if the API key/plan has no quota for
    image generation — as of mid-2026, Google's free Gemini API tier gives
    a quota of 0 requests/day for gemini-2.5-flash-image, so this requires
    a paid plan with billing enabled. Callers should catch this specifically
    and show a plain message rather than the raw API error.
    """
    client = _get_gemini_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("RESOURCE_EXHAUSTED", "429", "quota", "PERMISSION_DENIED")):
            raise ImageGenerationUnavailable(
                "Image generation isn't available on your current Gemini API plan. "
                "Free-tier keys currently have zero quota for image generation — "
                "this needs a paid Gemini API plan with billing enabled."
            ) from e
        raise

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates for this image prompt.")
    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None and inline_data.data:
            return inline_data.data
    raise RuntimeError("Gemini did not return image data for this prompt — try rephrasing it.")


# ----------------------------------------------------
# Video generation (Veo) — always via Gemini. Async: submit, poll, retrieve.
# ----------------------------------------------------
def generate_video(prompt: str) -> bytes:
    """Generate a video from a text prompt. BLOCKING — polls until the async
    job completes or times out. Returns raw MP4 bytes.

    Raises VideoGenerationUnavailable if the API key/plan can't reach Veo
    (this is gated by Google independently of this code — a paid/allowlisted
    tier may be required). Callers should catch this specifically and show a
    plain "not available on your plan" message rather than a generic error.
    """
    client = _get_gemini_client()

    try:
        operation = client.models.generate_videos(
            model=GEMINI_VIDEO_MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(number_of_videos=1, duration_seconds=5),
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("PERMISSION_DENIED", "403", "not allowed", "not enabled", "quota")):
            raise VideoGenerationUnavailable(
                "Video generation isn't available on your current Gemini API plan/key."
            ) from e
        raise

    waited = 0
    while not operation.done:
        if waited >= VIDEO_MAX_WAIT_SECONDS:
            raise TimeoutError(
                f"Video generation exceeded the {VIDEO_MAX_WAIT_SECONDS}s wait limit and timed out."
            )
        time.sleep(VIDEO_POLL_SECONDS)
        waited += VIDEO_POLL_SECONDS
        operation = client.operations.get(operation)

    generated = getattr(operation.response, "generated_videos", None) or []
    if not generated:
        raise RuntimeError("Gemini did not return a video for this prompt — try rephrasing it.")

    video_obj = generated[0].video
    video_bytes = getattr(video_obj, "video_bytes", None)
    if video_bytes:
        return video_bytes
    raise RuntimeError("Could not read generated video bytes from the API response.")


# ----------------------------------------------------
# Compatibility Wrapper
# ----------------------------------------------------
def get_llm():
    """Compatibility wrapper so app.py imports don't break."""
    return None
