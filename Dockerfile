# Hugging Face Spaces — Docker SDK
# Free CPU Basic tier: 2 vCPU / 16GB RAM, which comfortably fits
# TinyLlama (Q4_K_M, ~670MB) + the embedding model + the reranker.

FROM python:3.11-slim

# System dependencies:
# - tesseract-ocr, poppler-utils : OCR for scanned PDFs (pytesseract/pdf2image)
# - build-essential, cmake       : fallback compiler toolchain for
#                                  llama-cpp-python, only used if no
#                                  prebuilt wheel matches this platform
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user requirements.txt .

# Prefer prebuilt CPU wheels for llama-cpp-python (fast, no compile needed);
# pip automatically falls back to source (using the toolchain above) if no
# matching wheel is found for this platform/Python version.
RUN pip install --no-cache-dir --user -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY --chown=user . .

# Download the model weights at build time, not first request:
# faster/cleaner first response, and no runtime dependency on the
# Hugging Face Hub being reachable from inside the running container.
RUN python -c "from llm import _ensure_model_downloaded; _ensure_model_downloaded()"

ENV TOKENIZERS_PARALLELISM=false
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
