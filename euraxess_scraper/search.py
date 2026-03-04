from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import config
from .utils import clean_text


def _fts_query_text(query: str) -> str:
    tokens = [tok for tok in clean_text(query).split(" ") if tok]
    if not tokens:
        return ""
    # Phrase-like behavior per token to avoid special token parse errors.
    return " ".join([f'"{tok}"' for tok in tokens])


def query_fts(conn, query: str, limit: int = 100) -> list[tuple[str, int]]:
    text = _fts_query_text(query)
    if not text:
        return []
    try:
        rows = conn.execute(
            """
            SELECT j.job_id
            FROM jobs_fts
            JOIN jobs j ON jobs_fts.rowid = j.rowid
            WHERE jobs_fts MATCH ?
              AND j.http_status = 200
              AND j.cleaned_text IS NOT NULL
            ORDER BY bm25(jobs_fts)
            LIMIT ?
            """,
            (text, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(row["job_id"], idx + 1) for idx, row in enumerate(rows)]


def query_vector(
    query: str,
    *,
    limit: int = 100,
    index_dir: Path = config.INDEX_DIR,
    model_name: str = config.EMBEDDING_MODEL,
) -> list[tuple[str, int]]:
    index_path = index_dir / "faiss.index"
    map_path = index_dir / "faiss_mapping.json"
    if not index_path.exists() or not map_path.exists():
        return []

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    if not mapping:
        return []

    index = faiss.read_index(str(index_path))
    model = SentenceTransformer(model_name)
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    vector = np.asarray(vector, dtype="float32")

    scores, indices = index.search(vector, min(limit, len(mapping)))
    out: list[tuple[str, int]] = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0:
            continue
        out.append((mapping[idx], rank))
    return out


def rrf_merge(
    fts_ranked: list[tuple[str, int]],
    vector_ranked: list[tuple[str, int]],
    *,
    k: int = config.RRF_K,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for job_id, rank in fts_ranked:
        scores[job_id] = scores.get(job_id, 0.0) + 1.0 / (k + rank)
    for job_id, rank in vector_ranked:
        scores[job_id] = scores.get(job_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _metadata_for_jobs(conn, job_ids: list[str]) -> dict[str, dict]:
    if not job_ids:
        return {}
    placeholders = ",".join(["?"] * len(job_ids))
    rows = conn.execute(
        f"""
        SELECT job_id, title, organization, country, deadline, url
        FROM jobs
        WHERE job_id IN ({placeholders})
        """,
        tuple(job_ids),
    ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}


def hybrid_search(
    conn,
    *,
    query: str,
    top_k: int = 10,
    country: str | None = None,
    rrf_k: int = config.RRF_K,
    model_name: str = config.EMBEDDING_MODEL,
) -> list[dict]:
    fts_ranked = query_fts(conn, query, limit=100)
    vector_ranked = query_vector(query, limit=100, model_name=model_name)
    merged = rrf_merge(fts_ranked, vector_ranked, k=rrf_k)

    job_ids = [job_id for job_id, _ in merged]
    metadata = _metadata_for_jobs(conn, job_ids)

    out: list[dict] = []
    country_filter = country.lower() if country else None
    for job_id, score in merged:
        meta = metadata.get(job_id)
        if not meta:
            continue
        if country_filter and (meta.get("country") or "").lower() != country_filter:
            continue
        out.append(
            {
                "job_id": job_id,
                "title": meta.get("title"),
                "organization": meta.get("organization"),
                "country": meta.get("country"),
                "deadline": meta.get("deadline"),
                "url": meta.get("url"),
                "rrf_score": score,
            }
        )
        if len(out) >= top_k:
            break
    return out
