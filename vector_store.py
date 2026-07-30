"""
vector_store.py — FAISS-backed vector store with persistent, deterministic caching.

NOTE on the cache key: the caller (app.py) MUST build the `file_hash` from a
*sorted* list of per-file hashes. A Python set's iteration order is not
guaranteed to be stable across process restarts, so joining an unsorted set
produces a different hash on every redeploy and silently defeats caching.
"""
import logging
import os

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
VECTOR_DB_DIR = os.environ.get("VECTOR_DB_DIR", "vector_db")

# FIXED: previously the whole document was embedded in a single
# FAISS.from_documents() call. For a large PDF that can mean hundreds of
# chunks embedded at once, which spikes memory sharply and gives zero
# feedback until it's either done or the process is OOM-killed. Batching
# keeps peak memory much lower and, since each batch calls into the model
# separately, progress can be logged as it goes instead of one big black box.
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def create_vector_store(chunks, file_hash: str) -> FAISS:
    """Load a cached FAISS index for this exact set of documents, or build one."""
    embeddings = get_embeddings()
    db_path = os.path.join(VECTOR_DB_DIR, file_hash)

    if os.path.isdir(db_path):
        try:
            return FAISS.load_local(
                db_path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            logger.exception(
                "Cached FAISS index at %s could not be loaded, rebuilding", db_path
            )

    if not chunks:
        raise ValueError("No chunks to index — nothing to build a vector store from.")

    # FIXED: build the index in small batches instead of one giant call.
    # This lowers peak RAM usage (important on Streamlit Cloud's ~1GB free
    # tier) and logs progress so a genuinely slow build is visible in the
    # server logs instead of looking identical to a frozen/dead process.
    total = len(chunks)
    vector_store = None
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        logger.info("Embedding chunks %d-%d of %d", i + 1, min(i + EMBED_BATCH_SIZE, total), total)
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    try:
        vector_store.save_local(db_path)
    except Exception:
        # Non-fatal: keep serving from the in-memory index even if the disk
        # is read-only or the platform's storage is ephemeral.
        logger.exception(
            "Could not persist FAISS index to %s; continuing in-memory", db_path
        )

    return vector_store
