from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from . import config

LOGGER = logging.getLogger(__name__)


def ensure_fts_table(conn) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts
        USING fts5(title, cleaned_text, content=jobs, content_rowid=rowid)
        """
    )
    conn.commit()


def rebuild_fts(conn) -> int:
    ensure_fts_table(conn)
    try:
        conn.execute("DELETE FROM jobs_fts")
    except sqlite3.DatabaseError:
        # Recover from FTS virtual table corruption by recreating the index table.
        conn.execute("DROP TABLE IF EXISTS jobs_fts")
        ensure_fts_table(conn)
    conn.execute(
        """
        INSERT INTO jobs_fts(rowid, title, cleaned_text)
        SELECT rowid, COALESCE(title, ''), COALESCE(cleaned_text, '')
        FROM jobs
        WHERE http_status = 200
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        """
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) AS c FROM jobs_fts").fetchone()["c"]
    return int(count)


def _indexable_rows(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT rowid, job_id, title, cleaned_text
        FROM jobs
        WHERE http_status = 200
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        ORDER BY rowid
        """
    ).fetchall()
    return [dict(row) for row in rows]


def build_faiss_index(
    conn,
    *,
    model_name: str = config.EMBEDDING_MODEL,
    batch_size: int = 64,
    index_dir: Path = config.INDEX_DIR,
) -> dict:
    # Lazy imports keep non-index commands lightweight.
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    index_dir.mkdir(parents=True, exist_ok=True)
    rows = _indexable_rows(conn)

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    vectors_path = index_dir / "vectors.npy"

    mapping: list[str] = []
    vector_chunks: list[np.ndarray] = []
    if rows:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [f"{r.get('title') or ''}\n\n{r.get('cleaned_text') or ''}" for r in batch]
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype="float32")
            index.add(vectors)
            vector_chunks.append(vectors)
            mapping.extend([str(r["job_id"]) for r in batch])

    index_path = index_dir / "faiss.index"
    map_path = index_dir / "faiss_mapping.json"
    if vector_chunks:
        stacked = np.vstack(vector_chunks).astype("float32")
    else:
        stacked = np.zeros((0, dim), dtype="float32")

    faiss.write_index(index, str(index_path))
    map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(vectors_path, stacked)

    LOGGER.info("Built FAISS index: %s vectors", len(mapping))
    return {
        "vectors": len(mapping),
        "embedding_dim": dim,
        "index_path": str(index_path),
        "mapping_path": str(map_path),
        "vectors_path": str(vectors_path),
    }


def build_indexes(conn, *, model_name: str = config.EMBEDDING_MODEL, batch_size: int = 64) -> dict:
    fts_count = rebuild_fts(conn)
    faiss_info = build_faiss_index(conn, model_name=model_name, batch_size=batch_size)
    return {"fts_rows": fts_count, **faiss_info}


def index_status(index_dir: Path = config.INDEX_DIR) -> dict:
    map_path = index_dir / "faiss_mapping.json"
    idx_path = index_dir / "faiss.index"
    vectors_path = index_dir / "vectors.npy"
    vectors = 0
    if map_path.exists():
        try:
            vectors = len(json.loads(map_path.read_text(encoding="utf-8")))
        except Exception:
            vectors = 0
    return {
        "faiss_exists": idx_path.exists(),
        "mapping_exists": map_path.exists(),
        "vectors_matrix_exists": vectors_path.exists(),
        "vectors": vectors,
        "index_path": str(idx_path),
        "mapping_path": str(map_path),
        "vectors_path": str(vectors_path),
    }
