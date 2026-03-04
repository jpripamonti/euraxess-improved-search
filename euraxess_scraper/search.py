from __future__ import annotations

import json
import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config

LOGGER = logging.getLogger(__name__)
from .language import LANGUAGE_UNKNOWN
from .taxonomy import canonicalize_query, load_synonyms, normalize_job_type_filter
from .topics import TOPIC_OTHER, load_topic_domains, normalize_topic_filters
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
    except sqlite3.OperationalError as exc:
        LOGGER.warning("FTS query failed for %r: %s", text, exc)
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
        vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        vector = np.asarray(vector, dtype="float32")
        if matrix.shape[1] != vector.shape[0]:
            return []
    except Exception as exc:
        LOGGER.warning("Vector search failed: %s", exc)
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


def _candidate_pool_size(
    *,
    limit: int,
    offset: int,
    country: str | None,
    job_type: str | None,
    include_topics: list[str] | None,
    exclude_topics: list[str] | None,
    language: str | None = None,
) -> int:
    strictness = 0
    if country:
        strictness += 1
    if job_type:
        strictness += 1
    if include_topics:
        strictness += 1
    if exclude_topics:
        strictness += 1
    if language:
        strictness += 1
    multiplier = 12 + (strictness * 4)
    depth_boost = min(10, offset // max(1, limit))
    return max(600, offset + (limit * (multiplier + depth_boost)))


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
            job_type_score,
            topic_domain,
            topic_confidence,
            language,
            cleaned_text
        FROM jobs
        WHERE job_id IN ({placeholders})
        """,
        tuple(job_ids),
    ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}


def _infer_topic_hint_from_query(query: str, *, topics_path: str | None = None) -> str | None:
    text = clean_text(query).lower()
    if not text:
        return None
    domains = load_topic_domains(topics_path)
    best_domain: str | None = None
    best_score = 0
    for domain, meta in domains.items():
        if domain == TOPIC_OTHER:
            continue
        score = 0
        label = clean_text(str(meta.get("label") or "")).lower()
        if label and label in text:
            score += 2
        for seed in meta.get("seed_terms") or []:
            token = clean_text(str(seed)).lower()
            if token and token in text:
                score += 1
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def _candidate_rerank_text(meta: dict[str, Any]) -> str:
    title = clean_text(str(meta.get("title") or ""))
    org = clean_text(str(meta.get("organization") or ""))
    text = clean_text(str(meta.get("cleaned_text") or ""))[:2500]
    merged = "\n\n".join([chunk for chunk in [title, org, text] if chunk]).strip()
    return merged or title or org


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    model_name: str,
    rerank_weight: float = 0.35,
    rerank_top_n: int = 200,
    query_topic_hint: str | None = None,
    query_role_hint: str | None = None,
    debug: bool = False,
) -> list[dict[str, Any]]:
    if not query or len(candidates) < 2:
        return candidates
    try:
        import numpy as np
    except ImportError:
        return candidates

    head_size = min(len(candidates), max(10, int(rerank_top_n)))
    head = [dict(item) for item in candidates[:head_size]]
    tail = candidates[head_size:]

    texts = [_candidate_rerank_text(item["meta"]) for item in head]
    if not any(texts):
        return candidates
    try:
        embeddings = _load_embedding_model(model_name).encode(
            [query, *texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(embeddings, dtype="float32")
        query_vec = vectors[0]
        doc_vectors = vectors[1:]
        sims = np.asarray(doc_vectors @ query_vec, dtype="float32")
    except Exception as exc:
        LOGGER.warning("Reranking failed: %s", exc)
        return candidates

    q_norm = clean_text(query).lower()
    q_tokens = [tok for tok in q_norm.split() if tok]
    for idx, item in enumerate(head):
        meta = item["meta"]
        base_score = float(item.get("score") or 0.0)
        semantic_raw = float(sims[idx])
        semantic_score = max(0.0, min(1.0, (semantic_raw + 1.0) / 2.0))

        title_norm = clean_text(str(meta.get("title") or "")).lower()
        token_hits = sum(1 for token in q_tokens if token in title_norm)
        overlap = (token_hits / len(q_tokens)) if q_tokens else 0.0
        title_phrase = 0.12 if q_norm and len(q_norm) > 3 and q_norm in title_norm else 0.0
        title_boost = min(0.16, title_phrase + (0.08 * overlap))

        intent_boost = 0.0
        if query_role_hint and (meta.get("job_type_inferred") or "unknown") == query_role_hint:
            intent_boost += 0.05
        if query_topic_hint and (meta.get("topic_domain") or TOPIC_OTHER) == query_topic_hint:
            intent_boost += 0.05

        final_score = base_score + (semantic_score * float(rerank_weight)) + title_boost + intent_boost
        item["score"] = float(final_score)
        if debug:
            item["score_components"] = {
                "base_rrf": base_score,
                "semantic_score": semantic_score,
                "title_boost": title_boost,
                "intent_boost": intent_boost,
                "final_score": float(final_score),
            }

    head_sorted = sorted(head, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return head_sorted + tail


def _matches_filters(
    meta: dict[str, Any],
    *,
    country: str | None,
    job_type: str | None,
    include_topics: list[str] | None,
    exclude_topics: list[str] | None,
    language: str | None,
    active_only: bool,
    open_only: bool,
    now_iso: str,
) -> bool:
    if country and (meta.get("country") or "").lower() != country:
        return False
    if job_type:
        inferred = meta.get("job_type_inferred") or "unknown"
        if job_type == "other":
            # "other" filter = only actual Other Profession support-staff jobs
            if inferred != "other":
                return False
        elif job_type == "unknown":
            # "unknown" filter = unclassified / open-call jobs (null, empty, or 'unknown')
            if inferred != "unknown":
                return False
        elif inferred != job_type:
            return False
    topic_value = meta.get("topic_domain") or TOPIC_OTHER
    if include_topics and topic_value not in include_topics:
        return False
    if exclude_topics and topic_value in exclude_topics:
        return False
    if language and (meta.get("language") or LANGUAGE_UNKNOWN) != language:
        return False
    if active_only and meta.get("delisted_at"):
        return False
    deadline = meta.get("deadline")
    if open_only and deadline and deadline < now_iso:
        return False
    return True


def _row_to_result(
    meta: dict[str, Any],
    score: float | None,
    *,
    score_components: dict[str, Any] | None = None,
) -> dict:
    raw_text = clean_text(str(meta.get("cleaned_text") or ""))
    preview = raw_text[:1800]
    if raw_text and len(raw_text) > 1800:
        preview += " ..."
    row = {
        "job_id": meta.get("job_id"),
        "title": meta.get("title"),
        "organization": meta.get("organization"),
        "country": meta.get("country"),
        "deadline": meta.get("deadline"),
        "url": meta.get("url"),
        "job_type_inferred": meta.get("job_type_inferred") or "unknown",
        "job_type_score": meta.get("job_type_score"),
        "topic_domain": meta.get("topic_domain") or TOPIC_OTHER,
        "topic_confidence": meta.get("topic_confidence"),
        "language": meta.get("language") or LANGUAGE_UNKNOWN,
        "description_preview": preview,
        "rrf_score": score,
    }
    if score_components:
        row["score_components"] = score_components
    return row


def _build_where_clause(
    *,
    country: str | None,
    job_type: str | None,
    include_topics: list[str] | None,
    exclude_topics: list[str] | None,
    language: str | None,
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
        if job_type == "other":
            # "other" = actual Other Profession support-staff jobs
            clauses.append("job_type_inferred = 'other'")
        elif job_type == "unknown":
            # "unknown" = unclassified / open-call jobs (null, empty, or 'unknown')
            clauses.append(
                "(job_type_inferred IS NULL OR TRIM(job_type_inferred) = '' OR job_type_inferred = 'unknown')"
            )
        else:
            clauses.append("COALESCE(job_type_inferred, 'unknown') = ?")
            params.append(job_type)
    if include_topics:
        placeholders = ",".join(["?"] * len(include_topics))
        clauses.append(f"COALESCE(topic_domain, ?) IN ({placeholders})")
        params.append(TOPIC_OTHER)
        params.extend(include_topics)
    if exclude_topics:
        placeholders = ",".join(["?"] * len(exclude_topics))
        clauses.append(f"COALESCE(topic_domain, ?) NOT IN ({placeholders})")
        params.append(TOPIC_OTHER)
        params.extend(exclude_topics)
    if language:
        clauses.append("COALESCE(language, 'unknown') = ?")
        params.append(language)
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
    include_topics: list[str] | None,
    exclude_topics: list[str] | None,
    language: str | None,
    active_only: bool,
    open_only: bool,
) -> tuple[list[dict], int]:
    now_iso = now_utc_iso()
    where_sql, base_params = _build_where_clause(
        country=country,
        job_type=job_type,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        language=language,
        active_only=active_only,
        open_only=open_only,
        now_iso=now_iso,
    )

    count_row = conn.execute(f"SELECT COUNT(*) AS c FROM jobs WHERE {where_sql}", tuple(base_params)).fetchone()
    total = int(count_row["c"]) if count_row else 0

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
            job_type_score,
            topic_domain,
            topic_confidence,
            language,
            cleaned_text
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
    include_topics: list[str] | None,
    exclude_topics: list[str] | None,
    language: str | None,
    active_only: bool,
    open_only: bool,
    rrf_k: int,
    model_name: str,
    keyword_weight: float,
    vector_weight: float,
    semantic_only: bool,
    enable_rerank: bool,
    debug: bool,
    synonyms_path: str | None,
    topics_path: str | None,
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
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            language=language,
            active_only=active_only,
            open_only=open_only,
        )

    candidate_pool = _candidate_pool_size(
        limit=limit,
        offset=offset,
        country=country,
        job_type=job_type,
        include_topics=include_topics,
        exclude_topics=exclude_topics,
        language=language,
    )

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
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            language=language,
            active_only=active_only,
            open_only=open_only,
            now_iso=now_iso,
        ):
            continue
        filtered.append({"meta": meta, "score": float(score), "score_components": None})

    if enable_rerank:
        reranked = rerank_candidates(
            query=query_canonical,
            candidates=filtered,
            model_name=model_name,
            query_topic_hint=_infer_topic_hint_from_query(query_canonical, topics_path=topics_path),
            query_role_hint=normalize_job_type_filter(query_canonical, synonyms_path=synonyms_path),
            debug=debug,
        )
    else:
        reranked = filtered
    total = len(reranked)
    page = reranked[offset : offset + limit]
    rows = [
        _row_to_result(
            item["meta"],
            item.get("score"),
            score_components=item.get("score_components") if debug else None,
        )
        for item in page
    ]
    return rows, total


def hybrid_search(
    conn,
    *,
    query: str,
    top_k: int = 10,
    country: str | None = None,
    job_type: str | None = None,
    topic_domain: str | None = None,
    topic_domains: list[str] | None = None,
    exclude_topic_domains: list[str] | None = None,
    language: str | None = None,
    active_only: bool = True,
    open_only: bool = True,
    rrf_k: int = config.RRF_K,
    model_name: str = config.EMBEDDING_MODEL,
    keyword_weight: float = 1.0,
    vector_weight: float = 2.0,
    semantic_only: bool = False,
    enable_rerank: bool = True,
    debug: bool = False,
    limit: int | None = None,
    offset: int = 0,
    synonyms_path: str | None = None,
    topics_path: str | None = None,
    index_dir: Path | None = None,
) -> list[dict]:
    job_type_filter = normalize_job_type_filter(job_type, synonyms_path=synonyms_path)
    include_input = list(topic_domains or [])
    if topic_domain:
        include_input.append(topic_domain)
    include_filters = normalize_topic_filters(include_input, topics_path=topics_path)
    exclude_filters = normalize_topic_filters(exclude_topic_domains or [], topics_path=topics_path)
    if include_filters and exclude_filters:
        include_filters = [value for value in include_filters if value not in set(exclude_filters)]
    language_filter = language.lower().strip() if language and language not in {"all", "any", ""} else None
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
        include_topics=include_filters,
        exclude_topics=exclude_filters,
        language=language_filter,
        active_only=active_only,
        open_only=open_only,
        rrf_k=rrf_k,
        model_name=model_name,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        semantic_only=semantic_only,
        enable_rerank=enable_rerank,
        debug=debug,
        synonyms_path=synonyms_path,
        topics_path=topics_path,
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
    topic_domain: str | None = None,
    topic_domains: list[str] | None = None,
    exclude_topic_domains: list[str] | None = None,
    language: str | None = None,
    active_only: bool = True,
    open_only: bool = True,
    rrf_k: int = config.RRF_K,
    model_name: str = config.EMBEDDING_MODEL,
    keyword_weight: float = 1.0,
    vector_weight: float = 2.0,
    semantic_only: bool = False,
    enable_rerank: bool = True,
    debug: bool = False,
    synonyms_path: str | None = None,
    topics_path: str | None = None,
    index_dir: Path | None = None,
) -> dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(100, int(page_size)))
    offset = (page - 1) * page_size
    job_type_filter = normalize_job_type_filter(job_type, synonyms_path=synonyms_path)
    include_input = list(topic_domains or [])
    if topic_domain:
        include_input.append(topic_domain)
    include_filters = normalize_topic_filters(include_input, topics_path=topics_path)
    exclude_filters = normalize_topic_filters(exclude_topic_domains or [], topics_path=topics_path)
    if include_filters and exclude_filters:
        include_filters = [value for value in include_filters if value not in set(exclude_filters)]
    topic_filter = include_filters[0] if len(include_filters) == 1 else None
    language_filter = language.lower().strip() if language and language not in {"all", "any", ""} else None

    rows, total = _hybrid_jobs(
        conn,
        query=query,
        limit=page_size,
        offset=offset,
        country=(country.lower() if country else None),
        job_type=job_type_filter,
        include_topics=include_filters,
        exclude_topics=exclude_filters,
        language=language_filter,
        active_only=active_only,
        open_only=open_only,
        rrf_k=rrf_k,
        model_name=model_name,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        semantic_only=semantic_only,
        enable_rerank=enable_rerank,
        debug=debug,
        synonyms_path=synonyms_path,
        topics_path=topics_path,
        index_dir=index_dir,
    )

    has_next = (offset + len(rows)) < total
    payload: dict[str, Any] = {
        "query": query,
        "country": country,
        "job_type": job_type_filter or "all",
        "topic": topic_filter or "all",
        "include_topics": include_filters,
        "exclude_topics": exclude_filters,
        "language": language_filter or "all",
        "active_only": active_only,
        "open_only": open_only,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_next": has_next,
        "results": rows,
    }
    if debug:
        payload["debug"] = True
    return payload
