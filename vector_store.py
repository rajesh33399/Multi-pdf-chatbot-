"""
vector_store.py — FAISS-backed vector store with persistent, deterministic caching.

NOTE on the cache key: the caller (app.py) MUST build the `file_hash` from a
*sorted* list of per-file hashes. A Python set's iteration order is not
guaranteed to be stable across process restarts, so joining an unsorted set
produces a different hash on every redeploy and silently defeats caching.
"""
import logging
import os
from typing import Callable, Optional

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
VECTOR_DB_DIR = os.environ.get("VECTOR_DB_DIR", "vector_db")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))

ProgressCallback = Optional[Callable[[int, int], None]]


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def _embed_in_batches(chunks, embeddings, on_progress: ProgressCallback = None) -> FAISS:
    """Build a fresh FAISS index from `chunks`, in small batches, reporting
    progress as it goes so the caller can show real feedback instead of a
    spinner that looks identical whether it's working or dead."""
    total = len(chunks)
    vector_store = None
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
        done = min(i + EMBED_BATCH_SIZE, total)
        logger.info("Embedding chunks %d/%d", done, total)
        if on_progress:
            on_progress(done, total)
    return vector_store


def create_vector_store(chunks, file_hash: str, on_progress: ProgressCallback = None) -> FAISS:
    """Load a cached FAISS index for this exact set of documents, or build one
    from scratch. Used for the FIRST file(s) in a session, or when loading a
    fully-cached combination from a previous run."""
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

    vector_store = _embed_in_batches(chunks, embeddings, on_progress)

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    try:
        vector_store.save_local(db_path)
    except Exception:
        logger.exception(
            "Could not persist FAISS index to %s; continuing in-memory", db_path
        )

    return vector_store


def update_vector_store(
    existing_store: Optional[FAISS],
    new_chunks,
    combined_hash: str,
    on_progress: ProgressCallback = None,
) -> FAISS:
    """Add ONLY the newly-uploaded chunks to an already-built index, instead
    of re-embedding everything from scratch every time a file is added.

    This is the key fix: previously the app called create_vector_store()
    with the FULL combined chunk list on every upload, which meant adding a
    second file re-embedded the first file's chunks all over again. For a
    large PDF added on top of an existing index, that could double (or more)
    the work and make the app look frozen.
    """
    if existing_store is None:
        # First file(s) in this session — nothing to add to yet, build fresh.
        return create_vector_store(new_chunks, combined_hash, on_progress)

    if not new_chunks:
        return existing_store

    embeddings = get_embeddings()
    total = len(new_chunks)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = new_chunks[i : i + EMBED_BATCH_SIZE]
        existing_store.add_documents(batch)
        done = min(i + EMBED_BATCH_SIZE, total)
        logger.info("Embedding new chunks %d/%d", done, total)
        if on_progress:
            on_progress(done, total)

    db_path = os.path.join(VECTOR_DB_DIR, combined_hash)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    try:
        existing_store.save_local(db_path)
    except Exception:
        logger.exception(
            "Could not persist FAISS index to %s; continuing in-memory", db_path
        )

    return existing_store
