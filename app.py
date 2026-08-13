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

import fitz   # pymupdf — renders PDF pages to images for OCR fallback
import markdown as md_lib
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from pypdf import PdfReader
from supabase import create_client

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

# Initialize Supabase connection
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

if "user" not in st.session_state:
    st.session_state["user"] = None

# -------------------------------------------------------------------------
# LOGIN & SIGNUP GATE
# -------------------------------------------------------------------------
if not st.session_state["user"]:
    st.title("✨ SparkAI - Login")
    tab1, tab2 = st.tabs(["Log In", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state["user"] = res.user
                st.success("Logged in successfully!")
                st.rerun()
            except Exception as e:
                st.error("Invalid email or password.")

    with tab2:
        signup_email = st.text_input("Email", key="signup_email")
        signup_pass = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create Account"):
            try:
                supabase.auth.sign_up({"email": signup_email, "password": signup_pass})
                st.success("Account created! You can now log in.")
            except Exception as e:
                st.error(f"Sign up failed: {e}")

# -------------------------------------------------------------------------
# MAIN APP (Only visible after logging in)
# -------------------------------------------------------------------------
else:
    user_id = st.session_state["user"].id
    user_email = st.session_state["user"].email

    # Session state — each chat is its own entry: {title, messages, pinned,
    # mode, created, context}. `context` holds extracted text from any files
    # the user has uploaded into this chat, and gets passed to the LLM.
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
        st.markdown(_build_bubble_html(role, content), unsafe_allow_html=True)


    def open_assistant_media_row() -> None:
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
    }


    def _svg_icon(name: str, size: int = 18, color: str = "#5f6368") -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
                f'width="{size}" height="{size}" fill="{color}">{_ICON_SVGS[name]}</svg>')


    def _render_js_icon_button(icon_name: str, onclick_js: str, key: str, title: str,
                               color: str = "#5f6368", height: int = 34) -> None:
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


    def _speak_text(text: str, key: str) -> None:
        safe_text = json.dumps(text)
        components.html(
            f"<script>window.speechSynthesis.cancel();"
            f"window.speechSynthesis.speak(new SpeechSynthesisUtterance({safe_text}));</script>",
            height=0,
        )


    def _branch_chat(chat_id: str, idx: int) -> None:
        source = st.session_state.chats[chat_id]
        new_id = new_chat(source["mode"])
        branched = st.session_state.chats[new_id]
        branched["messages"] = [dict(m) for m in source["messages"][:idx + 1]]
        branched["context"] = source["context"]
        branched["uploaded_files"] = list(source["uploaded_files"])
        branched["title"] = (source["title"] + " (branch)")[:60]


    def render_user_message_actions(chat_id: str, idx: int, content: str) -> None:
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
        more_key = f"showmore_{chat_id}_{idx}"
        feedback = message.get("feedback")
        feedback_card_key = f"feedbackcard_{chat_id}_{idx}"
        speak_key = f"speaknow_{chat_id}_{idx}"

        if st.session_state.get(more_key):
            _menu_spacer, menu_col = st.columns([4, 13])
            with menu_col:
                with st.container(key=f"morecard_{chat_id}_{idx}"):
                    if st.button("Branch in new chat", icon=":material/call_split:",
                                 key=f"branchbtn_{chat_id}_{idx}", type="tertiary"):
                        _branch_chat(chat_id, idx)
                        st.session_state[more_key] = False
                        st.rerun()
                    if st.button("Listen", icon=":material/volume_up:",
                                 key=f"listenbtn_{chat_id}_{idx}", type="tertiary"):
                        st.session_state[speak_key] = True
                        st.session_state[more_key] = False
                        st.rerun()
                    if st.button("Report legal issue", icon=":material/flag:",
                                 key=f"reportbtn_{chat_id}_{idx}", type="tertiary"):
                        st.toast("Thanks — this has been reported.")
                        st.session_state[more_key] = False
                        st.rerun()

        if st.session_state.pop(speak_key, False):
            _speak_text(content, key=f"speak_{chat_id}_{idx}")

        col_up, col_down, col_regen, col_copy, col_more, _spacer = st.columns(
            [1, 1, 1, 1, 1, 12], gap="small"
        )

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
            if st.button(" ", icon=":material/more_horiz:", key=f"more_{chat_id}_{idx}",
                         type="tertiary", help="More"):
                st.session_state[more_key] = not st.session_state.get(more_key, False)
                st.rerun()

        if st.session_state.get(feedback_card_key):
            with st.container(key=f"feedbackcard_{chat_id}_{idx}"):
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
    OCR_FALLBACK_THRESHOLD = 20
    OCR_ZOOM = 2
    SUPPORTED_DOC_EXTENSIONS = (".pdf", ".txt", ".md", ".zip")
    SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


    def _ocr_pdf(raw: bytes) -> str:
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
        try:
            text_pymupdf = _extract_pdf_text_pymupdf(raw)
        except Exception:
            text_pymupdf = ""
        try:
            text_pypdf = _extract_pdf_text_pypdf(raw)
        except Exception:
            text_pypdf = ""

        text = text_pymupdf if len(text_pymupdf) >= len(text_pypdf) else text_pypdf

        if len(text) < OCR_FALLBACK_THRESHOLD:
            ocr_text = _ocr_pdf(raw)
            if ocr_text:
                return ocr_text

        return text


    def extract_text_from_bytes(name: str, raw: bytes) -> str:
        lower = name.lower()
        try:
            if lower.endswith(".pdf"):
                return _extract_pdf_text(raw)
            if lower.endswith((".txt", ".md")):
                return raw.decode("utf-8", errors="ignore").strip()
            if lower.endswith(SUPPORTED_IMAGE_EXTENSIONS):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                return pytesseract.image_to_string(img).strip()
            return ""
        except Exception:
            return ""


    def extract_zip_contents(raw_zip: bytes) -> tuple[list[tuple[str, str]], list[str]]:
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


    if not st.session_state.chats:
        new_chat("chat")

    # -------------------------------------------------------------------------
    # Styling
    # -------------------------------------------------------------------------
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: #1f1f1f; }
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
            padding-top: 10px;
        }
        [data-testid="stSidebar"] * { color: #1f1f1f !important; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #5f6368 !important; }
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
        [data-testid="stSidebar"] button:hover { background-color: #e8eaed !important; }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .sparkai-msg { display: flex; width: 100%; margin: 6px 0; }
        .sparkai-msg.sparkai-user { justify-content: flex-end; }
        .sparkai-msg.sparkai-user .sparkai-bubble {
            background-color: #f0f1f3;
            border-radius: 20px;
            padding: 10px 18px;
            max-width: 70%;
            color: #1f1f1f;
        }
        .sparkai-msg.sparkai-assistant { justify-content: flex-start; align-items: flex-start; gap: 8px; }
        .sparkai-msg.sparkai-assistant .sparkai-sparkle { font-size: 13px; line-height: 1.8; flex-shrink: 0; }
        .sparkai-msg.sparkai-assistant .sparkai-bubble { max-width: 85%; color: #1f1f1f; }
        .sparkai-bubble p:first-child { margin-top: 0; }
        .sparkai-bubble p:last-child { margin-bottom: 0; }
        .sparkai-bubble ol, .sparkai-bubble ul { margin: 6px 0; padding-left: 22px; }
        [data-testid="stChatInputFileUploadButton"] svg { display: none !important; }
        [data-testid="stChatInputFileUploadButton"]::before {
            content: "+";
            font-size: 22px;
            font-weight: 400;
            line-height: 1;
            color: #5f6368;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[class*="st-key-morecard_"],
        div[class*="st-key-feedbackcard_"] {
            background: #ffffff !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
            padding: 14px 18px !important;
            margin: 6px 0 !important;
        }
        div[class*="st-key-morecard_"] { max-width: 240px; }
        </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ✨ SparkAI")
        st.write(f"Logged in as: **{user_email}**")
        if st.button("Log Out", use_container_width=True):
            supabase.auth.sign_out()
            st.session_state["user"] = None
            st.rerun()

        st.markdown("---")

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
                                for m in chat_item["messages"]
                            )
                            st.toast("Conversation ready to copy!")
