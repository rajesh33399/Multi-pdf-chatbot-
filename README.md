 ✨ SparkAI — LLM-Powered Multimodal AI Assistant

SparkAI is an **LLM-powered multimodal AI assistant** built with **Python**, **Streamlit**, and **LangChain**. It combines general AI conversations with a **Retrieval-Augmented Generation (RAG)** pipeline for document-based question answering, along with OCR-powered processing for scanned documents and support for image-based interactions.

## 🚀 Key Features

### 💬 General AI Chat

- **Interactive AI Chat:** Engage in natural conversations through a clean, Gemini-inspired conversational interface.
- **Multiple LLM Providers:** Integrates **Google Gemini** and **Groq** APIs for AI-powered text generation.
- **Real-Time Streaming:** Displays AI responses progressively for a responsive conversational experience.
- **Context-Aware Conversations:** Maintains recent conversation history to provide more relevant responses.
- **Persistent Chat Sessions:** Manage multiple conversations within the application session.
- **Chat Management:** Create, rename, pin, delete, search, and branch conversations.
- **Message Editing:** Edit previous prompts and regenerate responses.
- **Response Regeneration:** Regenerate assistant responses when needed.
- **Transcript Exporting:** Export conversations for later use or documentation.
- **Text-to-Speech:** Listen to assistant responses using browser-based speech synthesis.

### 📚 Retrieval-Augmented Generation (RAG)

- **Document Question Answering:** Ask questions based on information contained in uploaded documents.
- **Multi-Format Document Support:** Process PDF, DOCX, TXT, and Markdown-based document content where supported.
- **Text Chunking:** Splits extracted document content into smaller chunks for efficient retrieval.
- **Embedding Generation:** Generates semantic embeddings using **Hugging Face Sentence Transformers**.
- **Vector Search:** Uses **FAISS** for semantic similarity search and document retrieval.
- **Context-Aware Responses:** Relevant document content is supplied to the LLM to generate document-aware answers.
- **Incremental Indexing:** Adds newly uploaded document chunks without unnecessarily re-processing previously indexed content.
- **Document Hashing:** Identifies previously processed files and reduces redundant embedding generation.
- **Persistent Vector Indexing:** Stores FAISS indexes locally for reuse.

### 📄 OCR & Document Processing

- **Scanned PDF Support:** Detects pages with insufficient native text extraction and applies OCR as a fallback.
- **Page-Level OCR:** OCR is applied only to pages that require it.
- **Scanned Document Support:** Extracts readable text from scanned documents.
- **Handwritten Document Support:** Attempts OCR extraction from handwritten pages using Tesseract.
- **OCR Fallback:** Uses native PDF extraction first and falls back to Tesseract when required.
- **Duplicate Chunk Removal:** Removes duplicate extracted chunks before embedding.

> **Note:** Tesseract OCR works best with printed or typed text. Handwritten or highly stylized documents may produce imperfect OCR results.

### 🖼️ Image Understanding

- **Image Input:** Upload images as part of a conversation.
- **Multimodal Processing:** Uses Google Gemini's multimodal capabilities to analyze uploaded images together with user questions.
- **Image-Based Questions:** Ask questions about the content of uploaded images.

### 🎨 Experimental Image Generation

- **Text-to-Image Capability:** SparkAI includes an experimental text-to-image feature that accepts natural-language prompts.
- **External Generation Service:** Image generation is handled through an external image-generation service.
- **Experimental Quality:** Generated results may vary depending on the prompt and external generation service.

> Image generation is currently treated as an experimental capability rather than a core document or conversational feature.

## 🛠️ Tech Stack

### Frontend & UI

- **Streamlit** — Interactive web interface and conversational UI
- **HTML / CSS** — Custom interface styling
- **JavaScript** — Client-side interactions and browser features

### Application & AI

- **Python** — Core application logic
- **LangChain** — LLM and RAG integration
- **Google Gemini API** — General and multimodal AI responses
- **Groq API** — Fast LLM inference

### RAG & Semantic Search

- **FAISS** — Vector storage and semantic similarity search
- **Hugging Face Sentence Transformers** — Document embedding generation
- **Incremental Indexing** — Avoids unnecessary document re-processing
- **Document Hashing** — Detects previously processed files
- **Batch Embedding** — Processes document chunks efficiently

### Document Processing

- **PyPDF / PyMuPDF** — PDF processing
- **python-docx** — DOCX document processing
- **Tesseract / pytesseract** — OCR for scanned documents
- **Pillow** — Image processing
- **Markdown** — Markdown content handling

### Deployment

- **Docker** — Application containerization
- **Streamlit** — Application serving and deployment

## 🏗️ Application Architecture

```text
                         ┌─────────────────────┐
                         │        User         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Streamlit UI    │
                         │  Chat + File Input  │
                         └──────────┬──────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
        ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
        │ General AI   │    │ Document RAG │    │    Images    │
        │    Chat      │    │    Pipeline  │    │ Understanding│
        └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
               │                   │                   │
               │                   ▼                   │
               │          ┌────────────────┐           │
               │          │    Document    │           │
               │          │    Loading     │           │
               │          └───────┬────────┘           │
               │                  │                    │
               │                  ▼                    │
               │          ┌────────────────┐           │
               │          │  Text / OCR    │           │
               │          │   Extraction   │           │
               │          └───────┬────────┘           │
               │                  │                    │
               │                  ▼                    │
               │          ┌────────────────┐           │
               │          │ Text Chunking  │           │
               │          └───────┬────────┘           │
               │                  │                    │
               │                  ▼                    │
               │          ┌────────────────┐           │
               │          │   Sentence     │           │
               │          │  Transformers  │           │
               │          └───────┬────────┘           │
               │                  │                    │
               │                  ▼                    │
               │          ┌────────────────┐           │
               │          │     FAISS      │           │
               │          │  Vector Store  │           │
               │          └───────┬────────┘           │
               │                  │                    │
               │                  ▼                    │
               │          ┌────────────────┐           │
               │          │    Semantic    │           │
               │          │    Retrieval   │           │
               │          └───────┬────────┘           │
               │                  │                    │
               └──────────────────┼────────────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │ LangChain / Prompt  │
                         │     Orchestration   │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
                ┌─────────────┐           ┌─────────────┐
                │    Groq     │           │   Gemini    │
                │     API     │           │     API     │
                └──────┬──────┘           └──────┬──────┘
                       │                         │
                       └────────────┬────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    AI Response      │
                         └─────────────────────┘

             













🔄 RAG Pipeline


Uploaded Document
        │
        ▼
Document Detection
        │
        ├──────────── PDF ─────────────┐
        │                              │
        ├──────────── DOCX ────────────┤
        │                              │
        └──────────── TXT / MD ────────┘
                                   │
                                   ▼
                            Text Extraction
                                   │
                                   ▼
                           Is Text Sufficient?
                               │         │
                              Yes        No
                               │         │
                               │         ▼
                               │    Tesseract OCR
                               │         │
                               └────┬────┘
                                    │
                                    ▼
                             Text Chunking
                                    │
                                    ▼
                         Sentence Transformer
                              Embeddings
                                    │
                                    ▼
                                  FAISS
                               Vector Index
                                    │
                                    ▼
                           Semantic Retrieval
                                    │
                                    ▼
                            Relevant Context
                                    │
                                    ▼
                             Gemini / Groq
                                    │
                                    ▼
                               AI Response








  ⚡ Performance Optimizations

SparkAI implements several optimizations to reduce unnec

SparkAI implements several optimizations to reduce unnecessary processing during document ingestion:

Incremental Indexing — Only newly uploaded content is processed when adding documents to an existing vector store.

Document Hashing — Previously processed documents can be identified without re-processing them.

Batch Embedding — Embeddings are generated in batches to better control CPU and memory usage.

Persistent FAISS Indexes — Vector indexes can be saved and reused.

Embedding Model Caching — The embedding model is cached during the application lifecycle.

Page-Level OCR Fallback — OCR is applied only to PDF pages where native extraction produces insufficient content.

Duplicate Chunk Removal — Repeated chunks are removed before indexing.





📁 Project Structure

      SparkAI/
│
├── .devcontainer/
│   └── # Development container configuration
│
├── .streamlit/
│   └── config.toml          # Streamlit theme and server configuration
│
├── .dockerignore            # Docker build exclusions
├── .gitignore               # Git ignored files
├── Dockerfile               # Docker container configuration
├── README.md                # Project documentation
│
├── app.py                   # Main Streamlit application
├── doc_loader.py            # DOCX and TXT document loading
├── llm.py                   # LLM and AI generation integrations
├── packages.txt             # System-level dependencies
├── pdf_loader.py            # PDF extraction and OCR processing
├── requirements.txt         # Python dependencies
├── runtime.txt              # Runtime configuration
├── text_splitter.py         # Document chunking and duplicate removal
└── vector_store.py          # FAISS embeddings, indexing and retrieval



⚙️ Getting Started
Prerequisites
Python 3.10+

Git

Tesseract OCR

Poppler

Google Gemini API key

Groq API key

Docker (optional)


Install Tesseract and Poppler
Ubuntu / Debian

sudo apt update
sudo apt install tesseract-ocr poppler-utils

macOS

brew install tesseract poppler

📥 Installation
1. Clone the Repository
Bash
git clone [https://github.com/rajesh33399/SparkAI.git](https://github.com/rajesh33399/SparkAI.git)
cd SparkAI
2. Create a Virtual Environment
Bash
python -m venv venv
3. Activate the Virtual Environment
Linux / macOS
Bash
source venv/bin/activate
Windows
Bash
venv\Scripts\activate
4. Install Dependencies
Bash
pip install -r requirements.txt
5. Configure API Keys
Configure your API keys using environment variables or Streamlit secrets.

Example:

Code snippet
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
Never commit API keys, tokens, or secret configuration files to GitHub.

6. Run SparkAI
Bash
streamlit run app.py
The application will be available at:

Plaintext
http://localhost:8501
🐳 Running with Docker
Build the Docker image:

Bash
docker build -t sparkai .
Run the container:

Bash
docker run -p 8501:8501 \
  -e GROQ_API_KEY="your_groq_api_key" \
  -e GEMINI_API_KEY="your_gemini_api_key" \
  sparkai
Then open:

Plaintext
http://localhost:8501
💡 Usage
General AI Chat
Open SparkAI.

Start a new chat.

Enter a question or prompt.

SparkAI sends the request to the configured LLM provider.

The response is streamed into the conversation.

Document-Based Question Answering
Upload a supported document.

SparkAI extracts the document content.

If a PDF page contains insufficient native text, OCR is applied.

The extracted content is split into chunks.

Sentence Transformer embeddings are generated.

FAISS stores the resulting vectors.

Ask a question about the uploaded document.

Relevant chunks are retrieved using semantic search.

The retrieved context is supplied to the LLM.

SparkAI generates a document-aware response.

Image Understanding
Upload an image through the chat interface.

Enter a question or instruction about the image.

SparkAI sends the image and prompt to the multimodal Gemini model.

The generated response is displayed in the conversation.

Experimental Image Generation
Open the image-generation capability.

Enter a natural-language image prompt.

SparkAI sends the prompt to the integrated external image-generation service.

The generated result is returned to the application.

Image generation is experimental and the quality or accuracy of generated images can vary depending on the prompt and external service.

📄 Supported Document Types
SparkAI currently supports document processing for:

.pdf

.docx

.txt

.md

ZIP-based workflows may also be handled by the application where supported.


🔐 Security

API credentials are loaded through environment variables or Streamlit secrets.

API keys should never be hard-coded into source files.

.env files and secret configuration files should remain excluded from Git.

Public repositories should never contain active API credentials.


⚠️ Limitations

AI response quality depends on the selected LLM and the quality of the retrieved context.

RAG responses depend on the quality of document extraction and semantic retrieval.

OCR accuracy can vary depending on scan quality, font, layout, and handwriting.

Tesseract is more reliable for printed text than complex handwriting.

Image generation is experimental and generated results may not always match the requested prompt precisely.

Free-tier API and hosting limits may affect response speed and availability.

Large or scanned documents may require additional processing time because OCR and embedding generation are CPU-intensive.



🚧 Future Improvements

🎬 Video Generation — Add text-to-video generation as a future multimodal capability.

🔎 Hybrid Search — Combine semantic vector search with keyword-based retrieval.

📊 Structured Data Analysis — Add CSV and Excel file analysis.

🌐 Web / URL Retrieval — Allow SparkAI to retrieve information from websites and URLs.

🎙️ Voice Interaction — Add speech-to-text and improved voice-based conversations.

👤 User Authentication — Add secure user accounts and authentication.

💾 Persistent Cloud Chat History — Store conversations across sessions.

☁️ Cloud Vector Database — Support scalable cloud-based vector storage.

📈 Improved Retrieval — Add reranking and more advanced retrieval strategies.

🔐 User-Level Data Isolation — Separate uploaded documents and chat data between users.


📌 Project Status

SparkAI is an actively developed LLM application that combines general AI chat, multimodal image understanding, and document-based RAG in a single Streamlit application.

The project focuses on integrating existing LLM and AI services rather than training a language model from scratch.

📄 License
Distributed under the MIT License. See the LICENSE file for more information.
