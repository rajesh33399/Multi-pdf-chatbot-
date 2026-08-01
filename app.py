"""
app.py — SparkAI: Gemini-style layout with multi-chat sessions, image
generation, and video generation.
"""
import time
import uuid

import streamlit as st

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
# mode, created}. This replaces the old single shared `messages` list, which
# meant clicking a "Recent" chat only relabeled the header without actually
# restoring that conversation's messages.
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


# Bootstrap the very first chat on cold start
if not st.session_state.chats:
    new_chat("chat")

# -------------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f1f1f; }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
        padding-top: 10px;
    }
    [data-testid="stSidebar"] button {
        text-align: left;
        justify-content: flex-start;
        border: none !important;
        background: transparent !important;
        font-weight: 400;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding: 4px 10px !important;
        min-height: 2.1rem !important;
        color: #1f1f1f !important;
    }
    [data-testid="stSidebar"] button p {
        color: #1f1f1f !important;
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
    /* Best-effort hover-to-open for a collapsed sidebar. This is additive
       only — it does NOT hide or replace Streamlit's native collapse
       arrow, so if this selector ever stops matching in a future Streamlit
       version, the manual arrow still works as a fallback. Relies on the
       aria-expanded attribute Streamlit sets on the sidebar container. */
    [data-testid="stSidebar"][aria-expanded="false"] {
        transition: margin-left 220ms ease-in-out;
    }
    [data-testid="stSidebar"][aria-expanded="false"]:hover {
        margin-left: 0 !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* NOTE: header is intentionally NOT hidden — it contains Streamlit's
       sidebar expand/collapse arrow. Hiding it makes the sidebar
       impossible to reopen once collapsed, with no way back. */
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

    # Build the visible list: filter by search query if active, pinned
    # chats first, then most-recent first. Only show chats that have at
    # least one message — an untouched fresh "New chat" doesn't clutter
    # Recents, matching how Gemini itself behaves.
    query = (st.session_state.get("search_query") or "").strip().lower()
    visible = [
        (cid, c) for cid, c in st.session_state.chats.items()
        if c["messages"] and (not query or query in c["title"].lower())
    ]
    visible.sort(key=lambda item: (not item[1]["pinned"], -item[1]["created"]))

    for cid, chat in visible:
        is_current = (cid == st.session_state.current_chat_id)

        if st.session_state.rename_target == cid:
            new_title = st.text_input(
                "Rename chat", value=chat["title"], key=f"rename_input_{cid}",
                label_visibility="collapsed"
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", key=f"rename_save_{cid}", use_container_width=True):
                    chat["title"] = new_title.strip() or chat["title"]
                    st.session_state.rename_target = None
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key=f"rename_cancel_{cid}", use_container_width=True):
                    st.session_state.rename_target = None
                    st.rerun()
            continue

        col_title, col_menu = st.columns([5, 1])
        with col_title:
            label = ("📌 " if chat["pinned"] else "") + chat["title"]
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
                            for m in chat["messages"] if m.get("type", "text") == "text"
                        )
                        st.text_area("Copy this transcript:", value=transcript,
                                     height=150, key=f"share_text_{cid}")
                    if st.button("📌 Unpin" if chat["pinned"] else "📌 Pin", key=f"pin_{cid}"):
                        chat["pinned"] = not chat["pinned"]
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
mode = chat["mode"]

mode_label = {"chat": "💬 Chat", "image": "🖼️ Image generation", "video": "🎬 Video generation"}[mode]
st.caption(mode_label)
st.markdown(f"### {chat['title']}")


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
            if message.get("attachment"):
                st.image(message["attachment"]["bytes"], width=200)
            st.markdown(message["content"])
        elif msg_type == "image":
            st.image(message["data"], caption=message.get("content"))
        elif msg_type == "video":
            st.video(message["data"])
        elif msg_type == "error":
            st.error(message["content"])

# ---- input + generation, branched by mode ----
if mode == "chat":
    prompt_data = st.chat_input(
        "Ask SparkAI", accept_file=True, file_type=["png", "jpg", "jpeg", "webp"],
    )
    if prompt_data:
        prompt = prompt_data.text or ""
        files = prompt_data.files or []
        attachment = None
        if files:
            f = files[0]
            attachment = {"bytes": f.getvalue(), "mime": f.type or "image/png", "name": f.name}

        if prompt or attachment:
            _maybe_set_title_from_prompt(prompt or f"📎 {attachment['name']}")

            user_msg = {"role": "user", "type": "text", "content": prompt}
            if attachment:
                user_msg["attachment"] = attachment
            chat["messages"].append(user_msg)

            with st.chat_message("user"):
                if attachment:
                    st.image(attachment["bytes"], width=200)
                if prompt:
                    st.markdown(prompt)

            text_history = [m for m in chat["messages"][:-1] if m.get("type", "text") == "text"]
            # If there's no text (image-only send), give the model something to act on.
            question = prompt or "Describe this image."

            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                stream_kwargs = {}
                if attachment:
                    stream_kwargs = {"image_bytes": attachment["bytes"], "image_mime_type": attachment["mime"]}
                for chunk in ask_llm_stream(context="", question=question, history=text_history, **stream_kwargs):
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
