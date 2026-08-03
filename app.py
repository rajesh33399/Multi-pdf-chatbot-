"""
app.py — SparkAI: Gemini-style layout with multi-chat sessions, image
generation, video generation, and document (PDF/ZIP/TXT) upload for RAG.
"""
import io
import time
import uuid
import zipfile

import streamlit as st
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


# -------------------------------------------------------------------------
# File parsing helpers
# -------------------------------------------------------------------------
def _extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


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
        return ""
    except Exception:
        return ""


def extract_zip_contents(raw_zip: bytes) -> tuple[list[tuple[str, str]], list[str]]:
    """Unzip in memory and extract text from every .pdf/.txt/.md entry inside.
    Returns (successes, failures) where successes is a list of
    (entry_name, text) and failures is a list of entry names that produced
    no text (corrupt, scanned-image PDF, unsupported type, etc.)."""
    successes, failures = [], []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if not name.lower().endswith((".pdf", ".txt", ".md")):
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

    /* ---- Hover-to-expand sidebar rail (Gemini-style) ---- */
    [data-testid="collapsedControl"] { display: none !important; }
    @media (min-width: 768px) {
        [data-testid="stSidebar"] {
            min-width: 72px !important;
            max-width: 72px !important;
            width: 72px !important;
            transition: min-width 0.18s ease, max-width 0.18s ease, width 0.18s ease;
            overflow-x: hidden;
            z-index: 999;
        }
        [data-testid="stSidebar"]:hover {
            min-width: 300px !important;
            max-width: 300px !important;
            width: 300px !important;
        }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            min-width: 260px;
        }
    }

    /* ---- Gemini-style chat bubbles ----
       We keep st.chat_message as the mechanism (so st.image/st.video keep
       working natively inside a message) and reskin it with CSS instead of
       hand-rolling HTML for every message: hide the colored avatar squares,
       right-align + pill-bubble the user's turn, left-align the assistant's
       turn with a small sparkle instead of an avatar block. These selectors
       (data-testid + the "Chat message from X" aria-label) come from
       Streamlit 1.60's actual shipped frontend, not guessed. */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarCustom"] {
        display: none !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 2px 0 !important;
        gap: 0 !important;
    }
    [data-testid="stChatMessage"][aria-label="Chat message from user"] {
        justify-content: flex-end !important;
    }
    [data-testid="stChatMessage"][aria-label="Chat message from user"] [data-testid="stChatMessageContent"] {
        background-color: #f0f1f3;
        border-radius: 20px;
        padding: 10px 18px !important;
        max-width: 70%;
        margin-left: auto;
    }
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] {
        justify-content: flex-start !important;
    }
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] [data-testid="stChatMessageContent"] {
        background: transparent;
        padding: 4px 0 4px 26px !important;
        max-width: 85%;
        position: relative;
    }
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] [data-testid="stChatMessageContent"]::before {
        content: "✨";
        position: absolute;
        left: 0;
        top: 2px;
        font-size: 13px;
    }

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
    with st.chat_message(message["role"]):
        if msg_type == "text":
            st.markdown(message["content"])
        elif msg_type == "image":
            st.image(message["data"], caption=message.get("content"))
        elif msg_type == "video":
            st.video(message["data"])
        elif msg_type == "error":
            st.error(message["content"])

# ---- input + generation, branched by mode ----
if mode == "chat":
    submission = st.chat_input(
        "Ask SparkAI",
        accept_file="multiple",
        file_type=["pdf", "txt", "md", "zip", "png", "jpg", "jpeg", "webp"],
    )

    if submission:
        prompt = (submission.text or "").strip()
        files = submission.files or []

        doc_files = [f for f in files if not f.type.startswith("image/")]
        image_files = [f for f in files if f.type.startswith("image/")]

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
                    failed.append(f"{f.name} (no .pdf/.txt/.md files found inside)")
            else:
                text = extract_text_from_bytes(f.name, raw)
                if text:
                    chat["context"] = (chat["context"] + f"\n\n--- {f.name} ---\n{text}").strip()
                    chat["uploaded_files"].append(f.name)
                    newly_attached.append(f.name)
                else:
                    failed.append(f.name)

        # Images go straight to the model as image data (not text-extracted).
        # This only actually reaches the model on the Gemini path — see
        # llm.py's ask_llm_stream, since Groq's chat model is text-only.
        pending_images = []
        for f in image_files:
            pending_images.append((f.read(), f.type))
            if f.name not in chat["uploaded_files"]:
                chat["uploaded_files"].append(f.name)
                newly_attached.append(f.name)

        if not prompt:
            prompt = "Summarize the attached file(s) and highlight anything important."

        _maybe_set_title_from_prompt(prompt)
        display_prompt = prompt
        if newly_attached:
            display_prompt += "\n\n📎 " + ", ".join(newly_attached)
        chat["messages"].append({"role": "user", "type": "text", "content": display_prompt})
        with st.chat_message("user"):
            st.markdown(display_prompt)
            for img_bytes, _mime in pending_images:
                st.image(img_bytes)

        if failed:
            st.warning(f"Couldn't extract text from: {', '.join(failed)} "
                       f"(likely a scanned/image-only PDF, an unsupported file type inside "
                       f"the zip, or a corrupt archive — OCR isn't wired up here).")

        text_history = [m for m in chat["messages"][:-1] if m.get("type", "text") == "text"]

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            for chunk in ask_llm_stream(
                context=chat["context"], question=prompt,
                history=text_history, images=pending_images or None,
            ):
                full_response += chunk
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)

        chat["messages"].append({"role": "assistant", "type": "text", "content": full_response})

elif mode == "image":
    if prompt := st.chat_input("Describe the image you want to generate"):
        _maybe_set_title_from_prompt(prompt)
        chat["messages"].append({"role": "user", "type": "text", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🎨 Generating image..."):
                try:
                    image_bytes = generate_image(prompt)
                    st.image(image_bytes, caption=prompt)
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
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🎬 Generating video — this can take a few minutes..."):
                try:
                    video_bytes = generate_video(prompt)
                    st.video(video_bytes)
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
