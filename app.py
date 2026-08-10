"""
app.py — SparkAI: Gemini-style layout with multi-chat sessions, image
generation, video generation, and document (PDF/ZIP/TXT) upload for RAG.
"""
import html
import io
import json
import time
import uuid
import zipfile

import fitz  # pymupdf — renders PDF pages to images for OCR fallback
import markdown as md_lib
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
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
if "editing_index" not in st.session_state:
    st.session_state.editing_index = {}  # {chat_id: message_index or None}
if "regenerate_index" not in st.session_state:
    st.session_state.regenerate_index = {}  # {chat_id: message_index or None}


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


_ICON_SVGS = {
    "thumb_up": '<path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2z"/>',
    "thumb_down": '<path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"/>',
    "refresh": '<path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>',
    "content_copy": '<path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>',
    "more_horiz": '<path d="M6 10c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm12 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-6 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>',
    "flag": '<path d="M14.4 6L14 4H5v17h2v-7h6.6l.4 2h7V6z"/>',
    "edit": '<path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>',
    "call_split": '<path d="M14 4l2.29 2.29-2.88 2.88 1.42 1.42 2.88-2.88L20 10V4h-6zM4 4v6l2.29-2.29 4.71 4.7V20h2v-8.41l-5.29-5.3L10 4H4z"/>',
    "volume_up": '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>',
}


def _svg_icon(name: str, size: int = 18, color: str = "#5f6368") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            f'width="{size}" height="{size}" fill="{color}">{_ICON_SVGS[name]}</svg>')


def _render_js_icon_button(icon_name: str, onclick_js: str, key: str, title: str,
                            color: str = "#5f6368", height: int = 34) -> None:
    """Renders one flat black icon button that runs REAL client-side JS on
    click. Uses st.components.v1.html — a genuine JS execution sandbox —
    rather than st.markdown(unsafe_allow_html=True), which was confirmed
    to mangle both multi-line HTML (Streamlit issue #859) and inline
    onclick handlers in earlier testing. Needed for anything that must
    happen instantly client-side with no visible extra UI (clipboard
    copy) or that drives a custom dropdown (see _render_more_menu)."""
    svg = _svg_icon(icon_name, color=color)
    html_code = f"""
    <html><head><style>
      html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
      #btn_{key} {{
        background:none; border:none; cursor:pointer; padding:6px;
        border-radius:6px; display:flex; align-items:center; justify-content:center;
        width:100%; height:100%;
      }}
      #btn_{key}:hover {{ background-color:#f0f1f3; }}
    </style></head>
    <body>
      <button id="btn_{key}" title="{title}">{svg}</button>
      <script>
        document.getElementById('btn_{key}').addEventListener('click', function() {{
            {onclick_js}
        }});
      </script>
    </body></html>
    """
    components.html(html_code, height=height)


def _render_copy_icon(text: str, key: str) -> None:
    """Instant clipboard copy — one click, nothing else visible, matching
    Gemini exactly (Gemini shows a small transient 'Prompt copied' toast;
    this flashes the icon to a blue checkmark for ~1s instead, entirely
    client-side so it's truly instant with no server round-trip)."""
    safe_text = json.dumps(text)
    check_svg = _svg_icon("content_copy", color="#1a73e8").replace("'", "\\'")
    onclick = (
        f"navigator.clipboard.writeText({safe_text}); "
        f"var b = document.getElementById('btn_{key}'); "
        f"var orig = b.innerHTML; "
        f"b.innerHTML = '{check_svg}'; "
        f"setTimeout(function(){{ b.innerHTML = orig; }}, 900);"
    )
    _render_js_icon_button("content_copy", onclick, key=key, title="Copy")


def _render_more_menu(text: str, chat_id: str, idx: int, key: str) -> None:
    """Gemini's '⋯' menu, matched to the reference video exactly: three
    items — Branch in new chat, Listen, Report legal issue. The trigger
    AND the dropdown panel are both self-contained in one JS sandbox
    (components.html), so opening/closing the menu itself needs no
    parent-DOM reaching. 'Branch' and 'Report' need real Python-side
    effects though (create a new chat / show a toast), so their clicks
    reach into the parent page and .click() a real, permanently-hidden
    st.button rendered right after this call (see render_assistant_
    message_actions) — found via a wrapper div ID, then `querySelector
    ('button')` inside it, since Streamlit's own generated button IDs
    aren't stable to target directly. 'Listen' is pure client-side text-
    to-speech (window.speechSynthesis) — no Python round-trip at all."""
    safe_text = json.dumps(text)
    branch_wrap_id = f"branchwrap_{chat_id}_{idx}"
    report_wrap_id = f"reportwrap_{chat_id}_{idx}"
    html_code = f"""
    <html><head><style>
      html, body {{ margin:0; padding:0; background:transparent; overflow:visible;
                    font-family:-apple-system,"Segoe UI",Roboto,sans-serif; }}
      #trigger_{key} {{
        background:none; border:none; cursor:pointer; padding:6px;
        border-radius:6px; display:flex; align-items:center; justify-content:center;
      }}
      #trigger_{key}:hover {{ background-color:#f0f1f3; }}
      #menu_{key} {{
        display:none; margin-top:2px; background:#fff; border-radius:8px;
        box-shadow:0 1px 3px rgba(0,0,0,0.3); width:210px; padding:6px 0;
      }}
      .mi_{key} {{
        display:flex; align-items:center; gap:12px; padding:8px 14px;
        cursor:pointer; font-size:14px; color:#1f1f1f;
      }}
      .mi_{key}:hover {{ background-color:#f0f1f3; }}
    </style></head>
    <body>
      <button id="trigger_{key}" title="More">{_svg_icon('more_horiz')}</button>
      <div id="menu_{key}">
        <div class="mi_{key}" id="branch_{key}">{_svg_icon('call_split', size=16)}<span>Branch in new chat</span></div>
        <div class="mi_{key}" id="listen_{key}">{_svg_icon('volume_up', size=16)}<span>Listen</span></div>
        <div class="mi_{key}" id="reportmi_{key}">{_svg_icon('flag', size=16)}<span>Report legal issue</span></div>
      </div>
      <script>
        var menu = document.getElementById('menu_{key}');
        document.getElementById('trigger_{key}').addEventListener('click', function() {{
            menu.style.display = (menu.style.display === 'none' || menu.style.display === '') ? 'block' : 'none';
        }});
        document.getElementById('branch_{key}').addEventListener('click', function() {{
            var wrap = window.parent.document.getElementById('{branch_wrap_id}');
            var btn = wrap ? wrap.querySelector('button') : null;
            if (btn) {{ btn.click(); }}
            menu.style.display = 'none';
        }});
        document.getElementById('listen_{key}').addEventListener('click', function() {{
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(new SpeechSynthesisUtterance({safe_text}));
            menu.style.display = 'none';
        }});
        document.getElementById('reportmi_{key}').addEventListener('click', function() {{
            var wrap = window.parent.document.getElementById('{report_wrap_id}');
            var btn = wrap ? wrap.querySelector('button') : null;
            if (btn) {{ btn.click(); }}
            menu.style.display = 'none';
        }});
      </script>
    </body></html>
    """
    components.html(html_code, height=170)


def _branch_chat(chat_id: str, idx: int) -> None:
    """Copies this chat's messages up to and including `idx` into a new
    chat and switches to it — mirrors Gemini's 'Branch in new chat'."""
    source = st.session_state.chats[chat_id]
    new_id = new_chat(source["mode"])
    branched = st.session_state.chats[new_id]
    branched["messages"] = [dict(m) for m in source["messages"][:idx + 1]]
    branched["context"] = source["context"]
    branched["uploaded_files"] = list(source["uploaded_files"])
    branched["title"] = (source["title"] + " (branch)")[:60]


def render_user_message_actions(chat_id: str, idx: int, content: str) -> None:
    """Copy + Edit row under a user message, matching Gemini's icon order
    and instant-copy behavior exactly."""
    _spacer, col_copy, col_edit = st.columns([14, 1, 1], gap="small")
    with col_copy:
        _render_copy_icon(content, key=f"copyuser_{chat_id}_{idx}")
    with col_edit:
        if st.button(" ", icon=":material/edit:", key=f"editbtn_{chat_id}_{idx}",
                      type="tertiary", help="Edit"):
            st.session_state.editing_index[chat_id] = idx
            st.rerun()


_BAD_RESPONSE_REASONS = [
    "Offensive/Unsafe", "Not factually correct", "Didn't follow instructions",
    "Personalisation issue", "More...", "Other",
]


def render_assistant_message_actions(chat_id: str, idx: int, content: str, message: dict) -> None:
    """👍 / 👎 / 🔁 / 📋 / ⋯ row under an assistant reply — icon set, order,
    and behavior matched to Gemini: thumbs-up just highlights; thumbs-down
    highlights AND opens a 'What went wrong?' card with reason pills
    (Gemini's actual bad-response flow, not just a color change); copy is
    instant client-side (no visible box); regenerate is its own icon
    (Gemini doesn't tuck it behind '⋯'); '⋯' opens a real dropdown with
    Branch in new chat / Listen / Report legal issue (see _render_more_
    menu) — matched to Gemini's own menu exactly, not the generic set."""
    col_up, col_down, col_regen, col_copy, col_more, _spacer = st.columns(
        [1, 1, 1, 1, 1, 12], gap="small"
    )
    feedback = message.get("feedback")
    feedback_card_key = f"feedbackcard_{chat_id}_{idx}"

    with col_up:
        if st.button(" ", icon=":material/thumb_up:", key=f"up_{chat_id}_{idx}",
                      type="primary" if feedback == "up" else "tertiary", help="Good response"):
            message["feedback"] = None if feedback == "up" else "up"
            st.session_state[feedback_card_key] = False
            st.rerun()
    with col_down:
        if st.button(" ", icon=":material/thumb_down:", key=f"down_{chat_id}_{idx}",
                      type="primary" if feedback == "down" else "tertiary", help="Bad response"):
            if feedback == "down":
                message["feedback"] = None
                st.session_state[feedback_card_key] = False
            else:
                message["feedback"] = "down"
                st.session_state[feedback_card_key] = True
            st.rerun()
    with col_regen:
        if st.button(" ", icon=":material/refresh:", key=f"regen_{chat_id}_{idx}",
                      type="tertiary", help="Regenerate response"):
            st.session_state.regenerate_index[chat_id] = idx
            st.rerun()
    with col_copy:
        _render_copy_icon(content, key=f"copyassistant_{chat_id}_{idx}")
    with col_more:
        _render_more_menu(content, chat_id, idx, key=f"more_{chat_id}_{idx}")

    # Permanently-hidden REAL buttons that the "⋯" menu's JS clicks
    # programmatically (via querySelector) for the two items that need a
    # genuine Python-side effect. Never shown to the user directly.
    st.markdown(f'<div id="branchwrap_{chat_id}_{idx}" style="display:none;">', unsafe_allow_html=True)
    if st.button(" ", key=f"branchbtn_{chat_id}_{idx}"):
        _branch_chat(chat_id, idx)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div id="reportwrap_{chat_id}_{idx}" style="display:none;">', unsafe_allow_html=True)
    if st.button(" ", key=f"reportbtn_{chat_id}_{idx}"):
        st.toast("Thanks — this has been reported.")
    st.markdown('</div>', unsafe_allow_html=True)

    # "What went wrong?" reason card — Gemini's actual bad-response flow,
    # shown right under the icon row when thumbs-down is active.
    if st.session_state.get(feedback_card_key):
        with st.container(border=True):
            col_title, col_close = st.columns([10, 1])
            with col_title:
                st.markdown("**What went wrong?**")
                st.caption("Your feedback helps improve SparkAI.")
            with col_close:
                if st.button(" ", icon=":material/close:", key=f"closefb_{chat_id}_{idx}",
                              type="tertiary", help="Close"):
                    st.session_state[feedback_card_key] = False
                    st.rerun()
            reason_cols = st.columns(len(_BAD_RESPONSE_REASONS))
            for reason_col, reason in zip(reason_cols, _BAD_RESPONSE_REASONS):
                with reason_col:
                    if st.button(reason, key=f"reason_{chat_id}_{idx}_{reason}", type="tertiary"):
                        st.session_state[feedback_card_key] = False
                        st.toast("Thanks for the detail — noted.")
                        st.rerun()


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

# Extensions this app knows how to read. Enforced in OUR Python code
# (see extract_text_from_bytes / extract_zip_contents) rather than via
# st.chat_input's built-in file_type client-side filter — that filter
# checks the browser-reported MIME type, which mobile browsers frequently
# report as empty/generic for files picked from apps like WhatsApp,
# Gallery, or Google Drive, causing valid .pdf/.jpg files to be rejected
# before they ever reach this server. Filtering by extension here instead
# means mobile uploads work the same as desktop.
SUPPORTED_DOC_EXTENSIONS = (".pdf", ".txt", ".md", ".zip")
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


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
        if lower.endswith(SUPPORTED_IMAGE_EXTENSIONS):
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
                    (".pdf", ".txt", ".md") + SUPPORTED_IMAGE_EXTENSIONS
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


def _stream_assistant_reply(prompt: str, images=None) -> None:
    """Streams a reply from the LLM and appends it to the current chat's
    message list. Shared by the normal chat_input submit path AND the
    'edit message' resend path (see the edit UI in the render loop below),
    so both regenerate a response the exact same way. Assumes the user's
    message has ALREADY been appended to chat["messages"] — history is
    built from everything before it."""
    text_history = [m for m in chat["messages"][:-1] if m.get("type", "text") == "text"]
    response_container = st.empty()
    full_response = ""
    for chunk in ask_llm_stream(
        context=chat["context"], question=prompt,
        history=text_history, images=images,
    ):
        full_response += chunk
        response_container.markdown(_build_bubble_html("assistant", full_response + "▌"), unsafe_allow_html=True)
    response_container.markdown(_build_bubble_html("assistant", full_response), unsafe_allow_html=True)
    chat["messages"].append({"role": "assistant", "type": "text", "content": full_response})


chat_id = st.session_state.current_chat_id

# ---- Handle a pending "regenerate response" request (from the assistant
# message's ⋯ menu) BEFORE rendering messages, so the replaced text shows
# immediately in the normal render loop below. Re-runs the same preceding
# user prompt through the model and replaces the reply IN PLACE (keeps its
# position in the conversation, clears any prior thumbs feedback on it). ----
_regen_idx = st.session_state.regenerate_index.get(chat_id)
if _regen_idx is not None:
    idx = _regen_idx
    if 0 < idx < len(chat["messages"]):
        user_msg = chat["messages"][idx - 1]
        if user_msg.get("type", "text") == "text" and user_msg["role"] == "user":
            history = [m for m in chat["messages"][:idx - 1] if m.get("type", "text") == "text"]
            with st.spinner("Regenerating..."):
                full_response = "".join(ask_llm_stream(
                    context=chat["context"], question=user_msg["content"], history=history,
                ))
            chat["messages"][idx]["content"] = full_response
            chat["messages"][idx]["feedback"] = None
    st.session_state.regenerate_index[chat_id] = None


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
editing_idx = st.session_state.editing_index.get(chat_id)

for idx, message in enumerate(chat["messages"]):
    msg_type = message.get("type", "text")

    if msg_type == "text" and message["role"] == "user" and idx == editing_idx:
        # Inline edit mode for this message — Gemini-style: rewrite the
        # prompt, then everything from here forward (this message and any
        # replies after it) is dropped and regenerated fresh.
        new_text = st.text_area(
            "Edit your message", value=message["content"],
            key=f"editarea_{chat_id}_{idx}", label_visibility="collapsed",
        )
        col_save, col_cancel, _spacer = st.columns([1, 1, 6])
        with col_save:
            if st.button("Send", key=f"editsave_{chat_id}_{idx}", type="primary"):
                new_text = new_text.strip()
                st.session_state.editing_index[chat_id] = None
                if new_text:
                    chat["messages"] = chat["messages"][:idx]
                    _maybe_set_title_from_prompt(new_text)
                    chat["messages"].append({"role": "user", "type": "text", "content": new_text})
                    _stream_assistant_reply(new_text)
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key=f"editcancel_{chat_id}_{idx}"):
                st.session_state.editing_index[chat_id] = None
                st.rerun()
        continue

    if msg_type == "text":
        render_message_bubble(message["role"], message["content"])
        if message["role"] == "user":
            render_user_message_actions(chat_id, idx, message["content"])
        else:
            render_assistant_message_actions(chat_id, idx, message["content"], message)
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
    # NOTE: no `file_type=[...]` here on purpose. st.chat_input's built-in
    # file_type filter validates against the browser-reported MIME type,
    # which mobile browsers (iOS Safari, Android Chrome/Samsung Internet)
    # frequently report as empty or generic for files coming from apps
    # like WhatsApp, Gallery, or Google Drive — that caused valid .pdf/
    # .jpg files to be rejected client-side before ever reaching this
    # server, even though desktop worked fine. We accept anything the
    # picker offers and instead filter by extension ourselves below via
    # extract_text_from_bytes / extract_zip_contents (SUPPORTED_DOC_
    # EXTENSIONS / SUPPORTED_IMAGE_EXTENSIONS) — genuinely unsupported
    # files still get skipped, just via our own code instead of a flaky
    # client-side MIME check.
    submission = st.chat_input(
        "Ask SparkAI",
        accept_file="multiple",
    )

    if submission:
        prompt = (submission.text or "").strip()
        files = submission.files or []

        # Classify by extension, not by the browser's reported MIME type
        # (submission.files[i].type) — same reasoning as above: mobile
        # browsers often report an empty/generic type for gallery/shared
        # files even though the filename extension is perfectly valid.
        def _is_image_file(f) -> bool:
            return f.name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)

        doc_files = [f for f in files if not _is_image_file(f)]
        image_files = [f for f in files if _is_image_file(f)]

        newly_attached, failed = [], []

        for f in doc_files:
            if f.name in chat["uploaded_files"]:
                continue
            raw = f.read()
            if f.name.lower().endswith(".zip"):
                successes, zip_failures = extract_zip_contents(raw)
                chat["uploaded_files"].append(f.name)
                for entry_name, text in successes:
                    label = f"{f.name}/{entry_name}"
                    chat["context"] = (chat["context"] + f"\n\n--- {label} ---\n{text}").strip()
                    newly_attached.append(label)
                failed.extend(f"{f.name}/{n}" for n in zip_failures)
                if not successes and not zip_failures:
                    failed.append(f"{f.name} (no supported files found inside)")
            elif f.name.lower().endswith((".pdf", ".txt", ".md")):
                text = extract_text_from_bytes(f.name, raw)
                if text:
                    chat["context"] = (chat["context"] + f"\n\n--- {f.name} ---\n{text}").strip()
                    chat["uploaded_files"].append(f.name)
                    newly_attached.append(f.name)
                else:
                    failed.append(f.name)
            else:
                # Extension not in our supported list at all — skip
                # cleanly instead of silently dropping it with no feedback.
                failed.append(f"{f.name} (unsupported file type)")

        # Images go straight to the model as image data (not text-extracted).
        # This only actually reaches the model on the Gemini path — see
        # llm.py's ask_llm_stream, since Groq's chat model is text-only.
        pending_images = []
        for f in image_files:
            if f.name in chat["uploaded_files"]:
                continue
            raw = f.read()
            mime = f.type or "image/jpeg"  # fall back if browser omitted MIME
            pending_images.append((raw, mime))
            chat["uploaded_files"].append(f.name)
            newly_attached.append(f.name)

        if not prompt:
            prompt = "Summarize the attached file(s) and highlight anything important."

        _maybe_set_title_from_prompt(prompt)
        display_prompt = prompt
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

        _stream_assistant_reply(prompt, images=pending_images or None)
        # Without this rerun, the just-added assistant message only shows
        # its action row (copy/edit/regenerate/etc) on the NEXT user
        # interaction — the streamed reply above is drawn inline during
        # THIS run, before it ever passes through the render loop that
        # attaches those buttons. Rerunning immediately routes it through
        # that same loop right away, so the icons appear instantly instead
        # of one message late (confirmed bug from user testing).
        st.rerun()

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
