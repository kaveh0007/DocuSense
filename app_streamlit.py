import os
import tempfile
from pathlib import Path
import streamlit as st
from backend.pdf_utils import extract_pdf_text, smart_chunk_pages
from backend.rag_faiss import FaissSQLiteStore
from backend.llm import generate_answer

st.set_page_config(page_title="RAG over PDFs", page_icon="📄", layout="wide")

st.title("📄🔍 RAG over PDFs — Ask your documents")

with st.sidebar:
    st.header("Settings")
    storage_dir = st.text_input("Storage directory", "storage")
    chunk_chars = st.number_input("Chunk size (chars)", min_value=300, max_value=4000, value=1000, step=100)
    overlap = st.number_input("Overlap (chars)", min_value=0, max_value=1000, value=150, step=10)
    top_k = st.slider("Top-K passages", min_value=1, max_value=10, value=5)
    model_name = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2", help="SBERT model path or HF id")

    st.markdown("---")
    st.caption("Optional: Set `OPENAI_API_KEY` in env for GPT-based answers.")

if "store" not in st.session_state:
    st.session_state.store = FaissSQLiteStore(storage_dir=storage_dir, model_name=model_name)

store = st.session_state.store

st.subheader("1) Upload and Ingest PDFs")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Ingest PDFs"):
    with st.spinner("Extracting and embedding..."):
        for up in uploaded_files:
            # Save to a temp file (and optionally persist to uploads/)
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            dest_path = uploads_dir / up.name
            with open(dest_path, "wb") as f:
                f.write(up.getbuffer())

            pages = extract_pdf_text(str(dest_path))
            chunks = smart_chunk_pages(pages, chunk_chars=int(chunk_chars), overlap=int(overlap))
            store.add_pdf_chunks(file_name=up.name, chunks=chunks)
        st.success("Ingestion complete!")

with st.expander("📚 Indexed Documents", expanded=False):
    docs = store.list_docs()
    if docs:
        st.write(docs)
    else:
        st.write("No documents ingested yet.")

st.subheader("2) Ask Questions")
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask a question about your PDFs...")
if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            contexts = store.query(user_q, top_k=top_k)
            if not contexts:
                st.markdown("_No relevant context found. Try ingesting PDFs or rephrasing your question._")
            else:
                answer = generate_answer(user_q, contexts)
                st.markdown(answer)

                with st.expander("Sources & Context"):
                    for i, c in enumerate(contexts, start=1):
                        st.markdown(f"**[{i}] {c['file_name']} — page {c['page']} (score: {c['score']:.3f})**")
                        st.markdown(f"> {c['text']}")

    st.rerun()
