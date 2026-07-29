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

    vector_store = FAISS.from_documents(chunks, embeddings)

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
