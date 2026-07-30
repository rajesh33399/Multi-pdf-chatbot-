"""
text_splitter.py — Chunks documents for embedding, with duplicate-chunk removal
(useful for cleaning up repeated OCR noise).
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    # Optimized chunk size and separators to prevent splitting code blocks, 
    # YAML files, JSON snippets, and multi-step lists apart.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    cleaned_chunks: list[Document] = []
    seen: set[str] = set()

    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        chunk.page_content = text
        cleaned_chunks.append(chunk)

    return cleaned_chunks
