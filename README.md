---
title: Chat With Multiple Documents
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Chat With Multiple Documents (Local RAG)

A local, self-hosted RAG chatbot: upload PDFs, Word docs, or text files and
ask questions across all of them. Runs entirely on CPU — no external API
keys required.

**Stack:** LangChain · FAISS · sentence-transformers embeddings ·
cross-encoder reranking (MMR + rerank) · TinyLlama-1.1B via
`llama-cpp-python` · Streamlit.

## Deploy to Hugging Face Spaces (free CPU Basic tier — 16GB RAM)

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **CPU basic** (free)
2. Push this folder to the Space's git repo:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
   git add .
   git commit -m "Initial deploy"
   git push origin main
   ```
3. The Space will build the Docker image (this downloads the ~670MB model
   during the build step — expect the first build to take a few minutes),
   then start automatically. No secrets or API keys needed.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

You'll also need `tesseract-ocr` and `poppler-utils` installed locally for
OCR on scanned PDFs (`apt install tesseract-ocr poppler-utils` on
Debian/Ubuntu, `brew install tesseract poppler` on macOS). The TinyLlama
model auto-downloads to `models/` on first run.

## Configuration

All tunables are environment variables (sensible defaults are baked in):

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_REPO_ID` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | HF repo for the GGUF model |
| `MODEL_FILENAME` | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Which quantization to use |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `RAG_TOP_K` | `5` | Chunks passed to the LLM as context |
| `RAG_FETCH_K` | `30` | Candidate pool size for MMR search |
| `RAG_MMR_LAMBDA` | `0.5` | 0 = max diversity, 1 = max relevance |
| `LLM_N_CTX` | `2048` | Model context window |
| `LLM_N_THREADS` | CPU count | Inference threads |

## Known limitations

- HF Spaces' free-tier disk is not guaranteed persistent across restarts —
  the FAISS cache and downloaded model may be rebuilt/re-downloaded after a
  Space restart. This doesn't affect correctness, only cold-start time.
- TinyLlama-1.1B is a small model; answer quality depends heavily on
  retrieval quality. If you outgrow it, swap `MODEL_REPO_ID`/`MODEL_FILENAME`
  for a larger GGUF model (adjust `LLM_N_CTX` accordingly).
