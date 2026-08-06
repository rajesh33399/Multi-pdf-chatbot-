"""
app.py — SparkAI: Gemini-style layout with multi-chat sessions, image
generation, video generation, and document (PDF/ZIP/TXT) upload for RAG.
"""
import html
import io
import time
import uuid
import zipfile

import fitz  # pymupdf — renders PDF pages to images for OCR fallback
import markdown as md_lib
import pytesseract
import streamlit as st
from PIL import Image
from pypdf import PdfReader

from llm import (
    ask_llm_stream, generate_image, generate_video,
    VideoGenerationUnavailable, ImageGenerationUnavailable,
)

st.set_page_config(
    page_title="SparkAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# Session state — each chat is its own entry: {title, messages, pinned,
# mode, created, context}. `context` holds extracted text from any files
# the user has uploaded into this chat, and gets passed to the LLM.
# -------------------------------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "show_search" not in st.session_state:
    st.session_state.show_search = False
if "rename_target" not in st.session_state:
    st.session_state.rename_target = None
if "confirm_delete_id" not in st.session_state:
    st.session_state.confirm_delete_id = None


def new_chat(mode: str = "chat") -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "pinned": False,
        "mode": mode,
        "created": time.time(),
        "context": "",
        "uploaded_files": [],
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


def get_current_chat() -> dict:
    cid = st.session_state.current_chat_id
    if cid is None or cid not in st.session_state.chats:
        cid = new_chat("chat")
    return st.session_state.chats[cid]


def switch_to(chat_id: str) -> None:
    st.session_state.current_chat_id = chat_id
    st.session_state.rename_target = None
    st.session_state.confirm_delete_id = None


def _build_bubble_html(role: str, content: str) -> str:
    body_html = md_lib.markdown(html.escape(content), extensions=["sane_lists"])
    if role == "user":
        return f'<div class="sparkai-msg sparkai-user"><div class="sparkai-bubble">{body_html}</div></div>'
    return (
        f'<div class="sparkai-msg sparkai-assistant">'
        f'<span class="sparkai-sparkle">✨</span>'
        f'<div class="sparkai-bubble">{body_html}</div></div>'
    )


def render_message_bubble(role: str, content: str) -> None:
    """Render one text message as a self-authored HTML bubble — right-aligned
    grey pill for the user, left-aligned plain text with a sparkle for the
    assistant. Content is HTML-escaped first (so a stray '<script>' typed by
    the user, or echoed back from an uploaded document, can't execute), then
    run through Markdown so bold/lists/links from the LLM still render."""
    st.markdown(_build_bubble_html(role, content), unsafe_allow_html=True)


def open_assistant_media_row() -> None:
    """Left-aligned sparkle row for wrapping st.image/st.video, which can't
    be embedded inside a markdown string. Streamlit renders each call as a
    sibling element in DOM order, so the widget in between lands inside
    this still-open div; close_assistant_media_row() closes it afterward."""
    st.markdown(
        '<div class="sparkai-msg sparkai-assistant">'
        '<span class="sparkai-sparkle">✨</span><div class="sparkai-bubble">',
        unsafe_allow_html=True,
    )


def close_assistant_media_row() -> None:
    st.markdown('</div></div>', unsafe_allow_html=True)


# -------------------------------------------------------------------------
# File parsing helpers
# -------------------------------------------------------------------------

# How short (in characters) the pypdf text-layer result has to be, across
# the WHOLE document, before we bother firing up OCR. Keeps normal
# text-based PDFs fast (no OCR needed) while still catching scanned /
# handwritten / image-only PDFs that have no real text layer at all.
OCR_FALLBACK_THRESHOLD = 20

# OCR render quality: 2x zoom ≈ 144 DPI, a good balance of accuracy vs
# speed. Bump to 3 if handwriting/small text is still coming out garbled.
OCR_ZOOM = 2


def _ocr_pdf(raw: bytes) -> str:
    """Render every page of a PDF to an image and run Tesseract OCR on it.
    Used when the PDF has no real text layer (scanned pages, handwritten
    notes exported as images, etc.) so text extraction alone returns
    nothing."""
    text_parts = []
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                text_parts.append(page_text.strip())
    finally:
        doc.close()
    return "\n".join(text_parts).strip()


def _extract_pdf_text_pymupdf(raw: bytes) -> str:
    """PyMuPDF's text extraction. Generally more reliable than pypdf for
    PDFs generated by design tools (Canva, Figma exports, some Word/Google
    Docs pipelines) that place text in a content-stream order that doesn't
    match the visual reading order — pypdf can silently drop or scramble
    sections (columns, sidebars, icon-adjacent text) in those files."""
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        parts = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(parts).strip()


def _extract_pdf_text_pypdf(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_pdf_text(raw: bytes) -> str:
    # Run both extractors and keep whichever pulled more content. Neither
    # is strictly "better" across all PDFs, but for a given file the one
    # that returns substantially less text is the one that dropped
    # sections — the longer result is the more complete one.
    try:
        text_pymupdf = _extract_pdf_text_pymupdf(raw)
    except Exception:
        text_pymupdf = ""
    try:
        text_pypdf = _extract_pdf_text_pypdf(raw)
    except Exception:
        text_pypdf = ""

    text = text_pymupdf if len(text_pymupdf) >= len(text_pypdf) else text_pypdf

    # Neither extractor found a usable text layer at all (scanned /
    # handwritten / image-only PDF) -> OCR as the last resort.
    if len(text) < OCR_FALLBACK_THRESHOLD:
        ocr_text = _ocr_pdf(raw)
        if ocr_text:
            return ocr_text

    return text


def extract_text_from_bytes(name: str, raw: bytes) -> str:
    """Best-effort text extraction from a filename + raw bytes. Returns ''
    (never raises) so one bad file can't crash the chat — callers decide
    how to surface a failure."""
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            return _extract_pdf_text(raw)
        if lower.endswith((".txt", ".md")):
            return raw.decode("utf-8", errors="ignore").strip()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            # A photo/scan of notes uploaded directly (not wrapped in a
            # PDF) -> OCR it the same way.
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return pytesseract.image_to_string(img).strip()
        return ""
    except Exception:
        return ""


def extract_zip_contents(raw_zip: bytes) -> tuple[list[tuple[str, str]], list[str]]:
    """Unzip in memory and extract text from every .pdf/.txt/.md/image entry
    inside. Returns (successes, failures) where successes is a list of
    (entry_name, text) and failures is a list of entry names that produced
    no text (corrupt, unreadable, unsupported type, etc.)."""
    successes, failures = [], []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if not name.lower().endswith(
                    (".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp")
                ):
                    continue
                try:
                    raw = zf.read(name)
                except Exception:
                    failures.append(name)
                    continue
                text = extract_text_from_bytes(name, raw)
                if text:
                    successes.append((name, text))
                else:
                    failures.append(name)
    except zipfile.BadZipFile:
        failures.append("(unreadable/corrupt zip archive)")
    return successes, failures


# Bootstrap the very first chat on cold start
if not st.session_state.chats:
    new_chat("chat")

# -------------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f1f1f; }

    /* ---- Sidebar base ---- */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e0e0e0;
        padding-top: 10px;
    }
    [data-testid="stSidebar"] * {
        color: #1f1f1f !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #5f6368 !important;
    }
    [data-testid="stSidebar"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        border: none !important;
        background: transparent !important;
        font-weight: 400;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding: 4px 10px !important;
        min-height: 2.1rem !important;
        width: 100%;
    }
    [data-testid="stSidebar"] button div,
    [data-testid="stSidebar"] button p,
    [data-testid="stSidebar"] button span {
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100%;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #e8eaed !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button {
        padding: 4px 4px !important;
    }
    [data-testid="stSidebar"] [data-testid="column"] {
        align-items: center;
    }
    /* Non-button sidebar text (the "SparkAI" heading, "Recent" caption)
       stays on one line and truncates instead of wrapping. */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ---- Gemini-style chat bubbles ----
       Self-authored markup (see render_message_bubble in app.py), not a
       reskin of Streamlit's internal chat_message DOM. The previous CSS
       reskin approach targeted real testids/aria-labels confirmed to exist
       in Streamlit's shipped frontend, but it still lost to Streamlit's
       own runtime style injection in production — rather than keep
       guessing at specificity/load-order blind, these classes are ours
       end to end, so what you see here is exactly what ships. */
    .sparkai-msg {
        display: flex;
        width: 100%;
        margin: 6px 0;
    }
    .sparkai-msg.sparkai-user {
        justify-content: flex-end;
    }
    .sparkai-msg.sparkai-user .sparkai-bubble {
        background-color: #f0f1f3;
        border-radius: 20px;
        padding: 10px 18px;
        max-width: 70%;
        color: #1f1f1f;
    }
    .sparkai-msg.sparkai-assistant {
        justify-content: flex-start;
        align-items: flex-start;
        gap: 8px;
    }
    .sparkai-msg.sparkai-assistant .sparkai-sparkle {
        font-size: 13px;
        line-height: 1.8;
        flex-shrink: 0;
    }
    .sparkai-msg.sparkai-assistant .sparkai-bubble {
        max-width: 85%;
        color: #1f1f1f;
    }
    .sparkai-bubble p:first-child { margin-top: 0; }
    .sparkai-bubble p:last-child { margin-bottom: 0; }
    .sparkai-bubble ol, .sparkai-bubble ul { margin: 6px 0; padding-left: 22px; }

    /* ---- Reskin the chat_input's built-in attach control to a plain
       "+" instead of the default paperclip, to match Gemini. This targets
       Streamlit's real stChatInputFileUploadButton testid; if a future
       Streamlit version renames it, this rule just silently no-ops and
       you get the default paperclip icon back, not a broken layout. ---- */
    [data-testid="stChatInputFileUploadButton"] svg {
        display: none !important;
    }
    [data-testid="stChatInputFileUploadButton"]::before {
        content: "+";
        font-size: 22px;
        font-weight: 400;
        line-height: 1;
        color: #5f6368;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✨ SparkAI")
    st.markdown("")

    if st.button("💬  New chat", key="btn_new", use_container_width=True):
        new_chat("chat")
        st.rerun()

    if st.button("🔍  Search chats", key="btn_search", use_container_width=True):
        st.session_state.show_search = not st.session_state.show_search
    if st.session_state.show_search:
        st.text_input("Search", key="search_query", label_visibility="collapsed",
                       placeholder="Search your chats...")

    if st.button("🖼️  Images", key="btn_images", use_container_width=True):
        new_chat("image")
        st.rerun()

    if st.button("🎬  Videos", key="btn_videos", use_container_width=True):
        new_chat("video")
        st.rerun()

    if st.button("📚  Library", key="btn_library", use_container_width=True):
        st.toast("Library view coming soon!")

    st.markdown("---")
    st.caption("Recent")

    query = (st.session_state.get("search_query") or "").strip().lower()
    visible = [
        (cid, c) for cid, c in st.session_state.chats.items()
        if c["messages"] and (not query or query in c["title"].lower())
    ]
    visible.sort(key=lambda item: (not item[1]["pinned"], -item[1]["created"]))

    for cid, chat_item in visible:
        is_current = (cid == st.session_state.current_chat_id)

        if st.session_state.rename_target == cid:
            new_title = st.text_input(
                "Rename chat", value=chat_item["title"], key=f"rename_input_{cid}",
                label_visibility="collapsed"
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", key=f"rename_save_{cid}", use_container_width=True):
                    chat_item["title"] = new_title.strip() or chat_item["title"]
                    st.session_state.rename_target = None
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key=f"rename_cancel_{cid}", use_container_width=True):
                    st.session_state.rename_target = None
                    st.rerun()
            continue

        col_title, col_menu = st.columns([5, 1])
        with col_title:
            label = ("📌 " if chat_item["pinned"] else "") + chat_item["title"]
            if st.button(label, key=f"open_{cid}", use_container_width=True,
                         type="secondary" if not is_current else "primary"):
                switch_to(cid)
                st.rerun()
        with col_menu:
            with st.popover("⋮", use_container_width=True):
                if st.session_state.confirm_delete_id == cid:
                    st.write("Delete this chat permanently?")
                    if st.button("Confirm delete", key=f"confirm_del_{cid}"):
                        del st.session_state.chats[cid]
                        st.session_state.confirm_delete_id = None
                        if st.session_state.current_chat_id == cid:
                            st.session_state.current_chat_id = None
                        st.rerun()
                else:
                    if st.button("🔗 Share conversation", key=f"share_{cid}"):
                        transcript = "\n\n".join(
                            f"{m['role'].capitalize()}: {m['content']}"
                            for m in chat_item["messages"] if m.get("type", "text") == "text"
                        )
                        st.text_area("Copy this transcript:", value=transcript,
                                     height=150, key=f"share_text_{cid}")
                    if st.button("📌 Unpin" if chat_item["pinned"] else "📌 Pin", key=f"pin_{cid}"):
                        chat_item["pinned"] = not chat_item["pinned"]
                        st.rerun()
                    if st.button("✏️ Rename", key=f"rename_{cid}"):
                        st.session_state.rename_target = cid
                        st.rerun()
                    if st.button("🗑️ Delete", key=f"delete_{cid}"):
                        st.session_state.confirm_delete_id = cid
                        st.rerun()

# -------------------------------------------------------------------------
# Main area
# -------------------------------------------------------------------------
chat = get_current_chat()
chat.setdefault("context", "")
chat.setdefault("uploaded_files", [])
mode = chat["mode"]

mode_label = {"chat": "💬 Chat", "image": "🖼️ Image generation", "video": "🎬 Video generation"}[mode]
st.caption(mode_label)
st.markdown(f"### {chat['title']}")

if chat["uploaded_files"]:
    st.caption("📎 Attached: " + ", ".join(chat["uploaded_files"]))


def _maybe_set_title_from_prompt(prompt: str) -> None:
    if chat["title"] == "New chat":
        chat["title"] = (prompt[:40] + "...") if len(prompt) > 40 else prompt


# ---- Gemini-style empty state for a brand-new, untouched chat ----
if not chat["messages"]:
    st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center;
                    justify-content:center; padding: 90px 0 60px 0; opacity: 0.85;">
            <div style="font-size: 48px; margin-bottom: 12px;">✨</div>
            <div style="font-size: 24px; color: #444;">Ready when you are</div>
        </div>
    """, unsafe_allow_html=True)


# ---- render existing messages ----
for message in chat["messages"]:
    msg_type = message.get("type", "text")
    if msg_type == "text":
        render_message_bubble(message["role"], message["content"])
    elif msg_type == "image":
        open_assistant_media_row()
        st.image(message["data"], caption=message.get("content"))
        close_assistant_media_row()
    elif msg_type == "video":
        open_assistant_media_row()
        st.video(message["data"])
        close_assistant_media_row()
    elif msg_type == "error":
        st.error(message["content"])

# ---- input + generation, branched by mode ----
if mode == "chat":
    chat.setdefault("processed_files", set())

    # Separate st.file_uploader instead of st.chat_input(accept_file=...).
    # chat_input's built-in attach flow has been unreliable on mobile
    # (files get flagged red / rejected regardless of size) — file_uploader
    # is Streamlit's oldest, most battle-tested upload widget and gives
    # actual error feedback instead of a silent red badge.
    with st.expander("📎 Attach files (PDF, TXT, MD, ZIP, images)", expanded=False):
        uploaded = st.file_uploader(
            "Attach files",
            type=["pdf", "txt", "md", "zip", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.current_chat_id}",
        )

    prompt = st.chat_input("Ask SparkAI")

    newly_attached, failed, pending_images = [], [], []

    for f in (uploaded or []):
        # Dedup against BOTH successes and prior failures so a failed file
        # doesn't get silently re-processed (and re-OCR'd) on every rerun —
        # file_uploader keeps returning the same files across reruns until
        # the user removes them, unlike chat_input's one-shot submission.
        if f.name in chat["processed_files"]:
            continue
        chat["processed_files"].add(f.name)

        raw = f.getvalue()
        if f.type.startswith("image/"):
            # Images go straight to the model as image data (not
            # text-extracted). This only actually reaches the model on the
            # Gemini path — see llm.py's ask_llm_stream, since Groq's chat
            # model is text-only.
            pending_images.append((raw, f.type))
            chat["uploaded_files"].append(f.name)
            newly_attached.append(f.name)
        elif f.name.lower().endswith(".zip"):
            successes, zip_failures = extract_zip_contents(raw)
            chat["uploaded_files"].append(f.name)
            for entry_name, text in successes:
                label = f"{f.name}/{entry_name}"
                chat["context"] = (chat["context"] + f"\n\n--- {label} ---\n{text}").strip()
                newly_attached.append(label)
            failed.extend(f"{f.name}/{n}" for n in zip_failures)
            if not successes and not zip_failures:
                failed.append(f"{f.name} (no supported files found inside)")
        else:
            text = extract_text_from_bytes(f.name, raw)
            if text:
                chat["context"] = (chat["context"] + f"\n\n--- {f.name} ---\n{text}").strip()
                chat["uploaded_files"].append(f.name)
                newly_attached.append(f.name)
            else:
                failed.append(f.name)

    # Fire a turn either when the user typed something and hit send, OR
    # when new files just landed (auto-summarize them, same as before).
    if prompt or newly_attached or failed:
        effective_prompt = (prompt or "").strip()
        if not effective_prompt:
            effective_prompt = "Summarize the attached file(s) and highlight anything important."

        _maybe_set_title_from_prompt(effective_prompt)
        display_prompt = effective_prompt
        if newly_attached:
            display_prompt += "\n\n📎 " + ", ".join(newly_attached)
        chat["messages"].append({"role": "user", "type": "text", "content": display_prompt})
        render_message_bubble("user", display_prompt)
        for img_bytes, _mime in pending_images:
            st.image(img_bytes, width=220)

        if failed:
            st.warning(f"Couldn't extract text from: {', '.join(failed)} "
                       f"(likely a corrupt file, an unsupported type inside the zip, "
                       f"or OCR couldn't read the image/handwriting clearly).")

        text_history = [m for m in chat["messages"][:-1] if m.get("type", "text") == "text"]

        response_container = st.empty()
        full_response = ""
        for chunk in ask_llm_stream(
            context=chat["context"], question=effective_prompt,
            history=text_history, images=pending_images or None,
        ):
            full_response += chunk
            response_container.markdown(_build_bubble_html("assistant", full_response + "▌"), unsafe_allow_html=True)
        response_container.markdown(_build_bubble_html("assistant", full_response), unsafe_allow_html=True)

        chat["messages"].append({"role": "assistant", "type": "text", "content": full_response})

elif mode == "image":
    if prompt := st.chat_input("Describe the image you want to generate"):
        _maybe_set_title_from_prompt(prompt)
        chat["messages"].append({"role": "user", "type": "text", "content": prompt})
        render_message_bubble("user", prompt)

        with st.spinner("🎨 Generating image..."):
            try:
                image_bytes = generate_image(prompt)
                open_assistant_media_row()
                st.image(image_bytes, caption=prompt)
                close_assistant_media_row()
                chat["messages"].append({
                    "role": "assistant", "type": "image",
                    "data": image_bytes, "content": prompt,
                })
            except ImageGenerationUnavailable as e:
                st.error(str(e))
                chat["messages"].append({"role": "assistant", "type": "error", "content": str(e)})
            except Exception as e:
                st.error(f"Couldn't generate that image: {e}")
                chat["messages"].append({
                    "role": "assistant", "type": "error",
                    "content": f"Couldn't generate that image: {e}",
                })

elif mode == "video":
    if prompt := st.chat_input("Describe the video you want to generate"):
        _maybe_set_title_from_prompt(prompt)
        chat["messages"].append({"role": "user", "type": "text", "content": prompt})
        render_message_bubble("user", prompt)

        with st.spinner("🎬 Generating video — this can take a few minutes..."):
            try:
                video_bytes = generate_video(prompt)
                open_assistant_media_row()
                st.video(video_bytes)
                close_assistant_media_row()
                chat["messages"].append({
                    "role": "assistant", "type": "video",
                    "data": video_bytes, "content": prompt,
                })
            except VideoGenerationUnavailable as e:
                st.error(str(e))
                chat["messages"].append({"role": "assistant", "type": "error", "content": str(e)})
            except Exception as e:
                st.error(f"Couldn't generate that video: {e}")
                chat["messages"].append({
                    "role": "assistant", "type": "error",
                    "content": f"Couldn't generate that video: {e}",
                })
