from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config
from .taxonomy import canonicalize_query, load_synonyms, normalize_job_type_filter
from .utils import clean_text, now_utc_iso


@lru_cache(maxsize=2)
def _load_embedding_model(name: str):
    from sentence_transformers import SentenceTransformer

    # Force CPU in query-time inference to avoid unstable MPS/meta tensor issues
    # under threaded web serving on some local environments.
    try:
        return SentenceTransformer(name, device="cpu")
    except TypeError:
        return SentenceTransformer(name)


def _fts_query_text(query: str, *, synonyms_path: str | None = None) -> str:
    tokens = [tok for tok in clean_text(query).split(" ") if tok]
    if not tokens:
        return ""
    aliases = load_synonyms(synonyms_path)
    groups: list[str] = []
    for token in tokens:
        if token in aliases:
            variants = [token, *[alias for alias in aliases[token] if alias != token]]
            quoted = [f'"{variant}"' for variant in variants]
            groups.append(f"({' OR '.join(quoted)})")
        else:
            groups.append(f'"{token}"')
    # Explicit AND keeps multi-term intent, while type synonyms expand as OR groups.
    return " AND ".join(groups)


def query_fts(
    conn,
    query: str,
    *,
    limit: int = 100,
    synonyms_path: str | None = None,
) -> list[tuple[str, int]]:
    text = _fts_query_text(query, synonyms_path=synonyms_path)
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
    index_dir: Path | None = None,
    model_name: str = config.EMBEDDING_MODEL,
) -> list[tuple[str, int]]:
    index_dir = index_dir or config.INDEX_DIR
    map_path = index_dir / "faiss_mapping.json"
    vectors_path = index_dir / "vectors.npy"
    if not map_path.exists() or not vectors_path.exists():
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    if not mapping:
        return []

    try:
        matrix = np.load(vectors_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return []
        model = _load_embedding_model(model_name)
        vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        vector = np.asarray(vector, dtype="float32")
        if matrix.shape[1] != vector.shape[0]:
            return []
    except Exception:
        return []

    k = min(int(limit), len(mapping), int(matrix.shape[0]))
    if k <= 0:
        return []

    scores = np.asarray(matrix @ vector, dtype="float32")
    if k == len(scores):
        indices = np.argsort(scores)[::-1]
    else:
        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

    out: list[tuple[str, int]] = []
    for rank, idx in enumerate(indices[:k], start=1):
        mapped_idx = int(idx)
        if mapped_idx < 0 or mapped_idx >= len(mapping):
            continue
        out.append((mapping[mapped_idx], rank))
    return out


def rrf_merge(
    fts_ranked: list[tuple[str, int]],
    vector_ranked: list[tuple[str, int]],
    *,
    k: int = config.RRF_K,
    keyword_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for job_id, rank in fts_ranked:
        scores[job_id] = scores.get(job_id, 0.0) + (keyword_weight / (k + rank))
    for job_id, rank in vector_ranked:
        scores[job_id] = scores.get(job_id, 0.0) + (vector_weight / (k + rank))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _metadata_for_jobs(conn, job_ids: list[str]) -> dict[str, dict]:
    if not job_ids:
        return {}
    placeholders = ",".join(["?"] * len(job_ids))
    rows = conn.execute(
        f"""
        SELECT
            job_id,
            title,
            organization,
            country,
            deadline,
            url,
            delisted_at,
            job_type_inferred,
            job_type_score
        FROM jobs
        WHERE job_id IN ({placeholders})
        """,
        tuple(job_ids),
    ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}


def _matches_filters(
    meta: dict[str, Any],
    *,
    country: str | None,
    job_type: str | None,
    active_only: bool,
    open_only: bool,
    now_iso: str,
) -> bool:
    if country and (meta.get("country") or "").lower() != country:
        return False
    if job_type and (meta.get("job_type_inferred") or "unknown") != job_type:
        return False
    if active_only and meta.get("delisted_at"):
        return False
    deadline = meta.get("deadline")
    if open_only and deadline and deadline < now_iso:
        return False
    return True


def _row_to_result(meta: dict[str, Any], score: float | None) -> dict:
    return {
        "job_id": meta.get("job_id"),
        "title": meta.get("title"),
        "organization": meta.get("organization"),
        "country": meta.get("country"),
        "deadline": meta.get("deadline"),
        "url": meta.get("url"),
        "job_type_inferred": meta.get("job_type_inferred") or "unknown",
        "job_type_score": meta.get("job_type_score"),
        "rrf_score": score,
    }


def _build_where_clause(
    *,
    country: str | None,
    job_type: str | None,
    active_only: bool,
    open_only: bool,
    now_iso: str,
) -> tuple[str, list[Any]]:
    clauses = [
        "http_status = 200",
        "cleaned_text IS NOT NULL",
        "TRIM(cleaned_text) != ''",
    ]
    params: list[Any] = []
    if country:
        clauses.append("LOWER(country) = ?")
        params.append(country)
    if job_type:
        clauses.append("COALESCE(job_type_inferred, 'unknown') = ?")
        params.append(job_type)
    if active_only:
        clauses.append("delisted_at IS NULL")
    if open_only:
        clauses.append("(deadline IS NULL OR deadline >= ?)")
        params.append(now_iso)
    return " AND ".join(clauses), params


def _direct_list_jobs(
    conn,
    *,
    limit: int,
    offset: int,
    country: str | None,
    job_type: str | None,
    active_only: bool,
    open_only: bool,
) -> tuple[list[dict], int]:
    now_iso = now_utc_iso()
    where_sql, base_params = _build_where_clause(
        country=country,
        job_type=job_type,
        active_only=active_only,
        open_only=open_only,
        now_iso=now_iso,
    )

    count_row = conn.execute(f"SELECT COUNT(*) AS c FROM jobs WHERE {where_sql}", tuple(base_params)).fetchone()
    total = int(count_row["c"]) if count_row else 0

    rows = conn.execute(
        f"""
        SELECT job_id, title, organization, country, deadline, url, delisted_at, job_type_inferred, job_type_score
        FROM jobs
        WHERE {where_sql}
        ORDER BY COALESCE(posted_date, deadline, last_seen_at) DESC, job_id DESC
        LIMIT ?
        OFFSET ?
        """,
        tuple(base_params + [limit, offset]),
    ).fetchall()

    return [_row_to_result(dict(row), None) for row in rows], total


def _hybrid_jobs(
    conn,
    *,
    query: str,
    limit: int,
    offset: int,
    country: str | None,
    job_type: str | None,
    active_only: bool,
    open_only: bool,
    rrf_k: int,
    model_name: str,
    keyword_weight: float,
    vector_weight: float,
    semantic_only: bool,
    synonyms_path: str | None,
    index_dir: Path | None,
) -> tuple[list[dict], int]:
    query_canonical = canonicalize_query(query, synonyms_path=synonyms_path)
    if not query_canonical:
        return _direct_list_jobs(
            conn,
            limit=limit,
            offset=offset,
            country=country,
            job_type=job_type,
            active_only=active_only,
            open_only=open_only,
        )

    candidate_pool = max(500, offset + (limit * 12))

    fts_ranked = [] if semantic_only else query_fts(
        conn,
        query_canonical,
        limit=candidate_pool,
        synonyms_path=synonyms_path,
    )
    vector_ranked = query_vector(
        query_canonical,
        limit=candidate_pool,
        model_name=model_name,
        index_dir=index_dir,
    )
    merged = rrf_merge(
        fts_ranked,
        vector_ranked,
        k=rrf_k,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
    )

    job_ids = [job_id for job_id, _ in merged]
    metadata = _metadata_for_jobs(conn, job_ids)
    country_filter = country.lower() if country else None
    now_iso = now_utc_iso()

    filtered: list[dict] = []
    for job_id, score in merged:
        meta = metadata.get(job_id)
        if not meta:
            continue
        if not _matches_filters(
            meta,
            country=country_filter,
            job_type=job_type,
            active_only=active_only,
            open_only=open_only,
            now_iso=now_iso,
        ):
            continue
        filtered.append(_row_to_result(meta, score))

    total = len(filtered)
    return filtered[offset : offset + limit], total


def hybrid_search(
    conn,
    *,
    query: str,
    top_k: int = 10,
    country: str | None = None,
    job_type: str | None = None,
    active_only: bool = True,
    open_only: bool = True,
    rrf_k: int = config.RRF_K,
    model_name: str = config.EMBEDDING_MODEL,
    keyword_weight: float = 1.0,
    vector_weight: float = 2.0,
    semantic_only: bool = False,
    limit: int | None = None,
    offset: int = 0,
    synonyms_path: str | None = None,
    index_dir: Path | None = None,
) -> list[dict]:
    job_type_filter = normalize_job_type_filter(job_type, synonyms_path=synonyms_path)
    limit = int(limit if limit is not None else top_k)
    limit = max(1, limit)
    offset = max(0, int(offset))

    rows, _ = _hybrid_jobs(
        conn,
        query=query,
        limit=limit,
        offset=offset,
        country=(country.lower() if country else None),
        job_type=job_type_filter,
        active_only=active_only,
        open_only=open_only,
        rrf_k=rrf_k,
        model_name=model_name,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        semantic_only=semantic_only,
        synonyms_path=synonyms_path,
        index_dir=index_dir,
    )
    return rows


def hybrid_search_page(
    conn,
    *,
    query: str,
    page: int = 1,
    page_size: int = 20,
    country: str | None = None,
    job_type: str | None = None,
    active_only: bool = True,
    open_only: bool = True,
    rrf_k: int = config.RRF_K,
    model_name: str = config.EMBEDDING_MODEL,
    keyword_weight: float = 1.0,
    vector_weight: float = 2.0,
    semantic_only: bool = False,
    synonyms_path: str | None = None,
    index_dir: Path | None = None,
) -> dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(100, int(page_size)))
    offset = (page - 1) * page_size
    job_type_filter = normalize_job_type_filter(job_type, synonyms_path=synonyms_path)

    rows, total = _hybrid_jobs(
        conn,
        query=query,
        limit=page_size,
        offset=offset,
        country=(country.lower() if country else None),
        job_type=job_type_filter,
        active_only=active_only,
        open_only=open_only,
        rrf_k=rrf_k,
        model_name=model_name,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        semantic_only=semantic_only,
        synonyms_path=synonyms_path,
        index_dir=index_dir,
    )

    has_next = (offset + len(rows)) < total
    return {
        "query": query,
        "country": country,
        "job_type": job_type_filter or "all",
        "active_only": active_only,
        "open_only": open_only,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_next": has_next,
        "results": rows,
    }
