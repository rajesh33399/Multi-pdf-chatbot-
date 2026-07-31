"""
app.py — Streamlit UI for the SparkChat Multi-Document RAG chatbot.
"""
import streamlit as st
import hashlib
import logging
import os
import re
import tempfile

# Must be set before sentence-transformers/tokenizers get imported anywhere
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sentence_transformers import CrossEncoder

from doc_loader import load_document
from llm import ask_llm_stream, get_llm
from text_splitter import split_documents
from vector_store import create_vector_store, update_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Config
# ----------------------------------------------------
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
FETCH_K = int(os.environ.get("RAG_FETCH_K", "30"))
MMR_LAMBDA = float(os.environ.get("RAG_MMR_LAMBDA", "0.5"))  # 0 = max diversity, 1 = max relevance
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "3500"))

SUPPORTED_TYPES = ["pdf", "docx", "txt"]

# 1. Page Configuration
st.set_page_config(
    page_title="SparkChat - Multi-Doc AI",
    page_icon="⚡",
    layout="wide"
)


# ----------------------------------------------------
# Cached / lazily-loaded resources
# ----------------------------------------------------
@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME)


@st.cache_resource(show_spinner="🤖 Loading local AI model (first run downloads ~650MB, please wait)...")
def warm_up_llm() -> bool:
    get_llm()
    return True


@st.cache_data(show_spinner=False)
def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


# ----------------------------------------------------
# Retrieval helpers
# ----------------------------------------------------
def get_filename_keywords(filename: str) -> list[str]:
    name = os.path.splitext(os.path.basename(filename))[0]
    words = re.split(r"[_\-\s.]+", name.lower())
    return [w for w in words if len(w) > 2]


def detect_target_documents(question: str, documents) -> set[str]:
    q = question.lower()
    tokens = set(re.findall(r"\b\w+\b", q))
    important_words = [w for w in tokens if len(w) > 3]

    source_preview: dict[str, str] = {}
    for doc in documents:
        source = doc.metadata.get("source")
        if not source:
            continue
        preview = source_preview.setdefault(source, "")
        if len(preview) < 1500:
            source_preview[source] += " " + doc.page_content.lower()

    matched = set()
    for source, preview in source_preview.items():
        filename_words = get_filename_keywords(source)
        if any(word in tokens for word in filename_words):
            matched.add(source)
            continue

        score = sum(1 for word in important_words if word in preview)
        if score >= 2:
            matched.add(source)

    return matched


def remove_duplicates(docs):
    seen = set()
    output = []
    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            output.append(doc)
    return output


def retrieve_documents(question, documents, vector_store, top_k=TOP_K):
    reranker = load_reranker()
    target_sources = detect_target_documents(question, documents)

    raw_results = vector_store.max_marginal_relevance_search(
        question, k=FETCH_K, fetch_k=FETCH_K * 3, lambda_mult=MMR_LAMBDA
    )

    if target_sources:
        filtered = [d for d in raw_results if d.metadata.get("source") in target_sources]
        if filtered:
            raw_results = filtered

    raw_results = remove_duplicates(raw_results)
    if not raw_results:
        return []

    pairs = [[question, doc.page_content] for doc in raw_results]
    scores = reranker.predict(pairs)

    scored = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": float(score),
        }
        for doc, score in zip(raw_results, scores)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def build_context(results, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    context = ""
    for item in results:
        block = f"[Source: {item['source']} | Page: {item['page']}]\n{item['text']}\n\n"
        if len(context) + len(block) > max_chars:
            break
        context += block
    return context


def render_sources(results) -> None:
    if not results:
        st.caption("No matching passages were found.")
        return
    st.caption("Scores are relative ranking scores from the reranker, not probabilities.")
    for i, item in enumerate(results, start=1):
        preview = item["text"][:300] + ("…" if len(item["text"]) > 300 else "")
        st.markdown(f"**{i}. {item['source']} — page {item['page']}** · match score {item['score']:.2f}")
        st.caption(preview)


# ----------------------------------------------------
# Session state initialization (Chats, Files, RAG state)
# ----------------------------------------------------
for key, default in [
    ("chats", {"New Conversation": []}),
    ("pinned_chats", []),
    ("current_chat", "New Conversation"),
    ("search_query", ""),
    ("renaming_chat", None),
    ("processed_files", set()),
    ("all_documents", []),
    ("all_chunks", []),
    ("vector_store", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------
# Sidebar — SparkChat Interface
# ----------------------------------------------------
with st.sidebar:
    # Top Logo and SparkChat Name Side-by-Side
    col_logo, col_title = st.columns([1, 3])
    with col_logo:
        # Ensure your Variation 1 logo image file is named this or change accordingly
        st.image("variation1_teal_logo.png", width=40)
    with col_title:
        st.markdown("<h3 style='margin: 0; padding-top: 4px;'>SparkChat</h3>", unsafe_allow_html=True)
    
    st.markdown("---")

    # 1. New chat button
    if st.button("✏️ New chat", use_container_width=True):
        new_name = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_name] = []
        st.session_state.current_chat = new_name
        st.rerun()

    # 2. Search chats input
    st.session_state.search_query = st.text_input("🔍 Search chats", placeholder="Search chats...", label_visibility="collapsed")

    st.markdown("---")
    
    # 3. Recent section with Pin & Three-Dot Menu Options
    st.markdown("**Recent**")

    # Handle rename state interface if triggered
    if st.session_state.renaming_chat:
        st.write(f"Renaming: *{st.session_state.renaming_chat}*")
        new_title = st.text_input("New name", value=st.session_state.renaming_chat, key="rename_input")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Save", use_container_width=True):
                if new_title and new_title != st.session_state.renaming_chat:
                    st.session_state.chats[new_title] = st.session_state.chats.pop(st.session_state.renaming_chat)
                    if st.session_state.current_chat == st.session_state.renaming_chat:
                        st.session_state.current_chat = new_title
                st.session_state.renaming_chat = None
                st.rerun()
        with col_r2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.renaming_chat = None
                st.rerun()
        st.markdown("---")

    # List recent chats with action buttons
    for chat_title in list(st.session_state.chats.keys()):
        if st.session_state.search_query.lower() in chat_title.lower():
            is_active = (chat_title == st.session_state.current_chat)
            is_pinned = chat_title in st.session_state.pinned_chats
            
            # Row layout for chat item name, pin indicator, and options menu
            c_chat, c_pin, c_menu = st.columns([0.62, 0.18, 0.20])
            
            with c_chat:
                btn_type = "primary" if is_active else "secondary"
                if st.button(chat_title, key=str(f"select_{chat_title}"), use_container_width=True, type=btn_type):
                    st.session_state.current_chat = chat_title
                    st.rerun()
            
            with c_pin:
                pin_label = "📌" if is_pinned else "📍"
                if st.button(pin_label, key=str(f"pin_{chat_title}"), help="Pin chat"):
                    if is_pinned:
                        st.session_state.pinned_chats.remove(chat_title)
                    else:
                        st.session_state.pinned_chats.append(chat_title)
                    st.rerun()
            
            with c_menu:
                # Popover acting as the three dots menu (...) containing Share, Rename, Pin, Archive, Delete
                with st.popover("⋮", help="More options"):
                    st.markdown(f"**{chat_title}**")
                    
                    if st.button("📤 Share", key=str(f"share_{chat_title}"), use_container_width=True):
                        st.toast(f"Link created for '{chat_title}'!")
                    
                    if st.button("✏️ Rename", key=str(f"rename_btn_{chat_title}"), use_container_width=True):
                        st.session_state.renaming_chat = chat_title
                        st.rerun()
                        
                    pin_text = "Unpin chat" if is_pinned else "Pin chat"
                    if st.button(f"📌 {pin_text}", key=str(f"pop_pin_{chat_title}"), use_container_width=True):
                        if is_pinned:
                            st.session_state.pinned_chats.remove(chat_title)
                        else:
                            st.session_state.pinned_chats.append(chat_title)
                        st.rerun()
                        
                    if st.button("📦 Archive", key=str(f"archive_{chat_title}"), use_container_width=True):
                        st.session_state.chats.pop(chat_title)
                        st.toast("Chat archived.")
                        st.rerun()
                        
                    if st.button("🗑️ Delete", key=str(f"delete_{chat_title}"), use_container_width=True):
                        st.session_state.chats.pop(chat_title)
                        if st.session_state.chats:
                            st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                        else:
                            st.session_state.chats["New Conversation"] = []
                            st.session_state.current_chat = "New Conversation"
                        st.rerun()

    # --- Document Upload & File Stats in Sidebar ---
    st.markdown("---")
    st.markdown("### 📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        new_files = False
        new_chunks_this_run = []

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_hash = get_file_hash(file_bytes)

            if file_hash in st.session_state.processed_files:
                continue

            new_files = True
            suffix = os.path.splitext(uploaded_file.name)[1] or ".bin"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(file_bytes)
                tmp_path = f.name

            try:
                with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                    try:
                        documents = load_document(tmp_path, uploaded_file.name)
                    except Exception as e:
                        logger.exception("Failed to load %s", uploaded_file.name)
                        st.error(f"Could not read {uploaded_file.name}: {e}")
                        continue

                    if not documents:
                        st.warning(f"No readable text found in {uploaded_file.name}")
                        continue

                    chunks = split_documents(documents)
                    if not chunks:
                        st.warning(f"No text found in {uploaded_file.name}")
                        continue

                    st.session_state.all_documents.extend(documents)
                    st.session_state.all_chunks.extend(chunks)
                    st.session_state.processed_files.add(file_hash)
                    new_chunks_this_run.extend(chunks)
            finally:
                os.unlink(tmp_path)

        if new_files and new_chunks_this_run:
            combined_hash = hashlib.sha256(
                "".join(sorted(st.session_state.processed_files)).encode()
            ).hexdigest()

            progress_bar = st.progress(0.0, text="🔄 Updating vector index...")

            def _on_progress(done: int, total: int) -> None:
                progress_bar.progress(done / total, text=f"🔄 Embedding chunks {done}/{total}...")

            try:
                st.session_state.vector_store = update_vector_store(
                    st.session_state.vector_store,
                    new_chunks_this_run,
                    combined_hash,
                    on_progress=_on_progress,
                )
                progress_bar.empty()
                st.success("✅ All documents processed successfully!")
            except Exception as e:
                progress_bar.empty()
                logger.exception("Vector index build failed")
                st.error(f"Failed to build the vector index: {e}")

    if st.session_state.processed_files:
        st.markdown("---")
        sources = sorted({doc.metadata.get("source", "?") for doc in st.session_state.all_documents})
        st.write("**Uploaded Files:**")
        for name in sources:
            st.write(f"• {name}")
        st.write(f"Pages/sections: {len(st.session_state.all_documents)}")
        st.write(f"Chunks: {len(st.session_state.all_chunks)}")

        if st.button("🗑 Clear All Documents", use_container_width=True):
            st.session_state.processed_files = set()
            st.session_state.all_documents = []
            st.session_state.all_chunks = []
            st.session_state.vector_store = None
            st.rerun()


# ----------------------------------------------------
# Main Chat Window Header
# ----------------------------------------------------
st.title("⚡ SparkChat")
st.caption("Chat with multiple documents using local AI & RAG.")

# Get current chat message history list
if st.session_state.current_chat not in st.session_state.chats:
    st.session_state.chats[st.session_state.current_chat] = []

current_messages = st.session_state.chats[st.session_state.current_chat]

# Render chat history for current session
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View sources"):
                render_sources(message["sources"])

# ----------------------------------------------------
# Chat Input & RAG Execution
# ----------------------------------------------------
question = st.chat_input("Ask something across your documents...")

if question:
    if st.session_state.vector_store is None:
        st.error("Please upload at least one document in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    current_messages.append({"role": "user", "content": question})

    history_for_llm = current_messages[:-1]

    with st.spinner("🔎 Searching documents..."):
        results = retrieve_documents(question, st.session_state.all_documents, st.session_state.vector_store)

    context = build_context(results)

    with st.chat_message("assistant"):
        if not context.strip():
            answer = "Information not found in uploaded documents."
            st.markdown(answer)
        else:
            with st.spinner("🤖 Loading local AI model (first question only, please wait)..."):
                warm_up_llm()
            answer = st.write_stream(ask_llm_stream(context, question, history=history_for_llm))

        if results:
            with st.expander("View sources"):
                render_sources(results)

    current_messages.append({"role": "assistant", "content": answer, "sources": results})

# --- Chat History Download Feature in Sidebar ---
if current_messages:
    chat_log = ""
    for msg in current_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_log += f"[{role}]: {msg['content']}\n\n"

    st.sidebar.download_button(
        label="📥 Download Current Chat",
        data=chat_log,
        file_name=f"{st.session_state.current_chat.replace(' ', '_')}_history.txt",
        mime="text/plain",
        use_container_width=True
    )
