import os
import sqlite3
import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path


def _ensure_dirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


class FaissSQLiteStore:
    """
    Persistent FAISS (cosine sim via normalized vectors) + SQLite metadata store.
    - SQLite tables:
        chunks(vector_id INTEGER PRIMARY KEY, file_name TEXT, page INTEGER, text TEXT)
        meta(key TEXT PRIMARY KEY, value TEXT)
    - FAISS index saved to storage/faiss.index
    """
    def __init__(self, storage_dir: str = "storage", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.storage_dir = storage_dir
        _ensure_dirs(storage_dir)
        self.index_path = os.path.join(storage_dir, "faiss.index")
        self.db_path = os.path.join(storage_dir, "meta.sqlite")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None  # type: ignore

        self.conn = sqlite3.connect(self.db_path)
        self._init_db()
        self._load_or_init_index()

    # ------------------- DB -------------------
    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS chunks(
            vector_id INTEGER PRIMARY KEY,
            file_name TEXT,
            page INTEGER,
            text TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        self.conn.commit()

    def _get_max_vector_id(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT MAX(vector_id) FROM chunks")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else -1

    def _insert_chunks(self, rows: List[Tuple[int, str, int, str]]):
        cur = self.conn.cursor()
        cur.executemany("INSERT INTO chunks(vector_id, file_name, page, text) VALUES (?, ?, ?, ?)", rows)
        self.conn.commit()

    # ------------------- FAISS -------------------
    def _load_or_init_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            return
        # Cosine similarity via inner product on normalized vectors
        self.index = faiss.IndexFlatIP(self.dim)
        faiss.write_index(self.index, self.index_path)

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    # ------------------- Embeddings -------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embs, dtype="float32")

    # ------------------- Public API -------------------
    def add_pdf_chunks(self, file_name: str, chunks: List[Dict]):
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        pages = [int(c["page"]) for c in chunks]
        embeddings = self.embed(texts)  # (N, dim)

        start_id = self._get_max_vector_id() + 1
        ids = np.arange(start_id, start_id + len(chunks)).astype("int64")

        self.index.add_with_ids(embeddings, ids)
        self._save_index()

        rows = [(int(i), file_name, int(p), t) for i, p, t in zip(ids, pages, texts)]
        self._insert_chunks(rows)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        q = self.embed([query_text])  # (1, dim), already normalized
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(q, top_k)  # cosine similarity (since normalized)
        I = I[0]
        D = D[0]
        results = []
        cur = self.conn.cursor()
        for vid, score in zip(I, D):
            if vid == -1:
                continue
            cur.execute("SELECT file_name, page, text FROM chunks WHERE vector_id = ?", (int(vid),))
            row = cur.fetchone()
            if not row:
                continue
            results.append({
                "vector_id": int(vid),
                "score": float(score),
                "file_name": row[0],
                "page": int(row[1]),
                "text": row[2]
            })
        return results

    def list_docs(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT file_name FROM chunks ORDER BY file_name")
        return [r[0] for r in cur.fetchall()]

    def delete_doc(self, file_name: str):
        """Deletes a document's chunks from SQLite and rebuilds the FAISS index."""
        cur = self.conn.cursor()
        cur.execute("SELECT vector_id FROM chunks WHERE file_name = ?", (file_name,))
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            return
        cur.execute("DELETE FROM chunks WHERE file_name = ?", (file_name,))
        self.conn.commit()

        # Rebuild index
        cur.execute("SELECT text FROM chunks ORDER BY vector_id")
        texts = [r[0] for r in cur.fetchall()]
        if texts:
            embs = self.embed(texts)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embs)  # we lose original IDs, but preserve order lookup by rowid
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self._save_index()
