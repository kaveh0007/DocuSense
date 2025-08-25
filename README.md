# DocuSense

DocuSense is an AI-powered app that reads your PDFs and answers questions using Retrieval-Augmented Generation (RAG).

**Tech Stack**
- **Python 3.9+**
- **Streamlit** for a simple ChatGPT-style UI
- **PyMuPDF** for robust PDF text extraction (with page numbers)
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) for embeddings
- **FAISS** (local) for fast vector search (cosine similarity)
- **OpenAI (optional)** for answer generation — auto-fallback to **FLAN‑T5 base** (open-source) if no API key found
- **SQLite** to persist metadata; FAISS index is saved on disk too

## Features
- Upload **multiple PDFs** and ingest them.
- **Chunking** by paragraphs with configurable sizes and overlap.
- **Persistent** embeddings + metadata across runs.
- **RAG** answers grounded in your PDFs only.
- **Citations**: shows file name, page number, and the exact context used.
- Handles **large PDFs** efficiently (streaming page-by-page).
- Error handling + validations.
- Bonus: Simple **CLI** for ingestion & querying. **Dockerfile** included.

## Quickstart

### 1) Create & activate a virtualenv
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> First run will download model weights (≈ 90MB for SBERT, ≈ 250MB for FLAN‑T5).

### 3) Run the Streamlit app
```bash
streamlit run app_streamlit.py
```

Then open the shown local URL in your browser.

### 4) (Optional) Use OpenAI for answer generation
Set `OPENAI_API_KEY` in your environment:
```bash
export OPENAI_API_KEY="sk-..."
```
If not set, the app will automatically use an open‑source fallback (FLAN‑T5) locally.

### 5) Try the CLI 
```bash
# Ingest one or more PDFs
python cli.py ingest sample_pdfs/sample.pdf

# Ask a question
python cli.py ask "What is this sample PDF about?"
```

## Project Structure
```
rag-pdf-ai/
├─ app_streamlit.py         # Streamlit chat UI (upload PDFs, ask questions)
├─ backend/
│  ├─ pdf_utils.py          # Parse PDFs, chunk text
│  ├─ rag_faiss.py          # FAISS + SQLite persistence
│  ├─ llm.py                # OpenAI or FLAN‑T5 answer generation
│  └─ __init__.py
├─ cli.py                   # Simple CLI to ingest & query
├─ requirements.txt
├─ Dockerfile
├─ sample_pdfs/
│  └─ sample.pdf
└─ storage/                 # Created at runtime (FAISS index + SQLite DB)
```

## Known Limitations
- Very large PDFs will ingest, but first-time embedding can take a while.
- If you rely on the open-source fallback (FLAN‑T5), answers are decent for many tasks but not as strong as GPT‑4 class models.
- Voice input is not included in this version (can be added using `streamlit-webrtc` + STT).

## License
MIT
