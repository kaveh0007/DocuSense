import argparse
from pathlib import Path
from backend.pdf_utils import extract_pdf_text, smart_chunk_pages
from backend.rag_faiss import FaissSQLiteStore
from backend.llm import generate_answer


def ingest(files, storage_dir="storage", chunk_chars=1000, overlap=150, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    store = FaissSQLiteStore(storage_dir=storage_dir, model_name=model_name)
    for f in files:
        pages = extract_pdf_text(str(f))
        chunks = smart_chunk_pages(pages, chunk_chars=chunk_chars, overlap=overlap)
        store.add_pdf_chunks(file_name=Path(f).name, chunks=chunks)
    print("Ingestion complete.")


def ask(question, storage_dir="storage", top_k=5, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    store = FaissSQLiteStore(storage_dir=storage_dir, model_name=model_name)
    contexts = store.query(question, top_k=top_k)
    if not contexts:
        print("No relevant context found.")
        return
    answer = generate_answer(question, contexts)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Sources ===\n")
    for i, c in enumerate(contexts, start=1):
        print(f"[{i}] {c['file_name']} — page {c['page']} (score: {c['score']:.3f})")
        print(c["text"][:500], "...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG over PDFs CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("files", nargs="+", help="PDF files to ingest")
    p_ing.add_argument("--storage_dir", default="storage")
    p_ing.add_argument("--chunk_chars", type=int, default=1000)
    p_ing.add_argument("--overlap", type=int, default=150)
    p_ing.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--storage_dir", default="storage")
    p_ask.add_argument("--top_k", type=int, default=5)
    p_ask.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(args.files, args.storage_dir, args.chunk_chars, args.overlap, args.model_name)
    elif args.cmd == "ask":
        ask(args.question, args.storage_dir, args.top_k, args.model_name)
    else:
        parser.print_help()
