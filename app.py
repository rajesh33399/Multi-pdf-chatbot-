"""
app.py — Streamlit UI for the Multi-Document RAG chatbot.
"""
import streamlit as st
import hashlib
import logging
import os
import re
import tempfile

# Must be set before sentence-transformers/tokenizers get imported anywhere
# in the dependency chain, or it logs a noisy (harmless) warning on every run.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sentence_transformers import CrossEncoder

from doc_loader import load_document
from llm import ask_llm_stream, get_llm
from text_splitter import split_documents
from vector_store import create_vector_store

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

st.set_page_config(page_title="Chat With Multiple Documents", page_icon="📚", layout="wide")


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


reranker = load_reranker()
warm_up_llm()


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
    """Heuristic filter: does the question name a specific file, or match one
    file's content much more strongly than the others? Generic — works for
    any filenames/content, not tied to any particular document type."""
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
    """MMR search for diverse, non-redundant candidates, optional source
    filtering, then cross-encoder reranking for final relevance ordering."""
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
# App title
# ----------------------------------------------------
st.title("📚 Chat With Multiple Documents")
st.caption("Upload PDFs, Word docs, or text files and ask questions across all of them.")


# ----------------------------------------------------
# Session state
# ----------------------------------------------------
for key, default in [
    ("processed_files", set()),
    ("all_documents", []),
    ("all_chunks", []),
    ("vector_store", None),
    ("messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------
# Upload
# ----------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if uploaded_files:
    new_files = False

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
                documents = load_document(tmp_path, uploaded_file.name)

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
        finally:
            os.unlink(tmp_path)

    if new_files and st.session_state.all_chunks:
        with st.spinner("🔄 Updating vector index..."):
            # Sort before joining: a set's iteration order isn't guaranteed
            # stable across process restarts, so this keeps the cache key
            # (and therefore the on-disk cache hit) deterministic.
            combined_hash = hashlib.sha256(
                "".join(sorted(st.session_state.processed_files)).encode()
            ).hexdigest()
            st.session_state.vector_store = create_vector_store(
                st.session_state.all_chunks, combined_hash
            )
        st.success("✅ All documents processed successfully!")


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
if st.session_state.processed_files:
    st.sidebar.markdown("### 📁 Uploaded Files")
    sources = sorted({doc.metadata.get("source", "?") for doc in st.session_state.all_documents})
    for name in sources:
        st.sidebar.write(f"• {name}")

    st.sidebar.write(f"Pages/sections: {len(st.session_state.all_documents)}")
    st.sidebar.write(f"Chunks: {len(st.session_state.all_chunks)}")

    if st.sidebar.button("🗑 Clear All"):
        st.session_state.processed_files = set()
        st.session_state.all_documents = []
        st.session_state.all_chunks = []
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------
# Chat history (persisted, including sources, across reruns)
# ----------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View sources"):
                render_sources(message["sources"])


# ----------------------------------------------------
# Chat input
# ----------------------------------------------------
question = st.chat_input("Ask something about your uploaded documents")

if question:
    if st.session_state.vector_store is None:
        st.error("Please upload at least one document first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    history_for_llm = st.session_state.messages[:-1]  # everything before this question

    with st.spinner("🔎 Searching documents..."):
        results = retrieve_documents(question, st.session_state.all_documents, st.session_state.vector_store)

    context = build_context(results)

    with st.chat_message("assistant"):
        if not context.strip():
            answer = "Information not found in uploaded documents."
            st.markdown(answer)
        else:
            answer = st.write_stream(ask_llm_stream(context, question, history=history_for_llm))

        if results:
            with st.expander("View sources"):
                render_sources(results)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
    # --- Chat History Download Feature ---
if "messages" in st.session_state and st.session_state.messages:
    # 1. Format the conversation list into clean text lines
    chat_log = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_log += f"[{role}]: {msg['content']}\n\n"
    
    # 2. Add a styled download button 
    st.sidebar.download_button(
        label="📥 Download Chat History",
        data=chat_log,
        file_name="chat_history.txt",
        mime="text/plain"
    )
