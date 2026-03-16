from __future__ import annotations

import sqlite3
from pathlib import Path

from euraxess_scraper import db, indexing, search

MISSING_INDEX_DIR = Path("/tmp/euraxess_missing_index_for_tests_only")


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    return conn


def _insert_job(conn, **overrides):
    record = {column: None for column in db.JOB_COLUMNS}
    record.update(
        {
            "job_id": overrides.get("job_id", "1"),
            "url": overrides.get("url", f"https://euraxess.ec.europa.eu/jobs/{overrides.get('job_id', '1')}"),
            "title": overrides.get("title", "Job"),
            "organization": overrides.get("organization", "Org"),
            "country": overrides.get("country", "Germany"),
            "cleaned_text": overrides.get("cleaned_text", "sample description"),
            "sections_json": "{}",
            "content_hash": overrides.get("content_hash", "hash"),
            "first_seen_at": "2026-03-01T00:00:00Z",
            "last_seen_at": "2026-03-01T00:00:00Z",
            "fetched_at": "2026-03-01T00:00:00Z",
            "http_status": 200,
            "job_type_inferred": overrides.get("job_type_inferred", "unknown"),
            "job_type_score": overrides.get("job_type_score", 50),
            "topic_domain": overrides.get("topic_domain", "other"),
            "topic_confidence": overrides.get("topic_confidence", 0.5),
            "deadline": overrides.get("deadline", "2099-01-01T00:00:00Z"),
            "delisted_at": overrides.get("delisted_at"),
        }
    )
    db.upsert_job_detail(conn, record)


def test_strict_type_filter_is_applied():
    conn = _conn()
    _insert_job(conn, job_id="10", title="Postdoc Role", job_type_inferred="postdoc")
    _insert_job(conn, job_id="11", title="PhD Role", job_type_inferred="phd")

    rows = search.hybrid_search(
        conn,
        query="",
        job_type="phd",
        active_only=False,
        open_only=False,
        limit=20,
        index_dir=MISSING_INDEX_DIR,
    )

    assert rows
    assert {row["job_type_inferred"] for row in rows} == {"phd"}


def test_postdoc_query_matches_postdoctoral_variant():
    conn = _conn()
    _insert_job(
        conn,
        job_id="15",
        title="Postdoctoral Research Fellow",
        job_type_inferred="postdoc",
        cleaned_text="Postdoctoral position in molecular biology.",
    )
    _insert_job(
        conn,
        job_id="16",
        title="PhD Candidate",
        job_type_inferred="phd",
        cleaned_text="Doctoral training position.",
    )
    indexing.rebuild_fts(conn)

    rows = search.hybrid_search(
        conn,
        query="postdoc",
        active_only=False,
        open_only=False,
        limit=20,
        index_dir=MISSING_INDEX_DIR,
    )

    ids = {row["job_id"] for row in rows}
    assert "15" in ids


def test_default_filters_exclude_closed_and_delisted():
    conn = _conn()
    _insert_job(conn, job_id="20", title="Open Postdoc", job_type_inferred="postdoc")
    _insert_job(
        conn,
        job_id="21",
        title="Expired PhD",
        job_type_inferred="phd",
        deadline="2000-01-01T00:00:00Z",
    )
    _insert_job(
        conn,
        job_id="22",
        title="Delisted Professor",
        job_type_inferred="professor",
        delisted_at="2026-03-02T00:00:00Z",
    )

    rows = search.hybrid_search(conn, query="", limit=50, index_dir=MISSING_INDEX_DIR)
    ids = {row["job_id"] for row in rows}

    assert ids == {"20"}


def test_all_mode_can_return_multiple_job_types():
    conn = _conn()
    _insert_job(conn, job_id="30", title="Open Postdoc", job_type_inferred="postdoc")
    _insert_job(conn, job_id="31", title="Open PhD", job_type_inferred="phd")
    _insert_job(conn, job_id="32", title="Open Professor", job_type_inferred="professor")

    rows = search.hybrid_search(
        conn,
        query="",
        active_only=False,
        open_only=False,
        limit=50,
        index_dir=MISSING_INDEX_DIR,
    )
    job_types = {row["job_type_inferred"] for row in rows}

    assert {"postdoc", "phd", "professor"}.issubset(job_types)
    assert all("description_preview" in row for row in rows)


def test_strict_topic_filter_is_applied():
    conn = _conn()
    _insert_job(
        conn,
        job_id="40",
        title="PhD in Sociology",
        job_type_inferred="phd",
        topic_domain="social_sciences",
    )
    _insert_job(
        conn,
        job_id="41",
        title="PhD in Quantum Physics",
        job_type_inferred="phd",
        topic_domain="natural_sciences",
    )

    rows = search.hybrid_search(
        conn,
        query="",
        topic_domain="social_sciences",
        active_only=False,
        open_only=False,
        limit=20,
        index_dir=MISSING_INDEX_DIR,
    )

    assert rows
    assert {row["topic_domain"] for row in rows} == {"social_sciences"}


def test_multi_topic_include_filter_supports_two_domains():
    conn = _conn()
    _insert_job(
        conn,
        job_id="42",
        title="PhD in Sociology",
        job_type_inferred="phd",
        topic_domain="social_sciences",
    )
    _insert_job(
        conn,
        job_id="43",
        title="General Research Position",
        job_type_inferred="unknown",
        topic_domain="other",
    )
    _insert_job(
        conn,
        job_id="44",
        title="PhD in Quantum Physics",
        job_type_inferred="phd",
        topic_domain="natural_sciences",
    )

    rows = search.hybrid_search(
        conn,
        query="",
        topic_domains=["social_sciences", "other"],
        active_only=False,
        open_only=False,
        limit=50,
        index_dir=MISSING_INDEX_DIR,
    )

    assert rows
    assert {row["topic_domain"] for row in rows} == {"social_sciences", "other"}


def test_exclude_topic_filter_removes_selected_categories():
    conn = _conn()
    _insert_job(conn, job_id="45", title="Social Position", topic_domain="social_sciences")
    _insert_job(conn, job_id="46", title="Other Position", topic_domain="other")
    _insert_job(conn, job_id="47", title="STEM Position", topic_domain="natural_sciences")

    rows = search.hybrid_search(
        conn,
        query="",
        topic_domains=["social_sciences", "other", "natural_sciences"],
        exclude_topic_domains=["other"],
        active_only=False,
        open_only=False,
        limit=50,
        index_dir=MISSING_INDEX_DIR,
    )

    assert rows
    assert {row["topic_domain"] for row in rows} == {"social_sciences", "natural_sciences"}


def test_query_role_hint_does_not_override_strict_job_filter():
    conn = _conn()
    _insert_job(conn, job_id="50", title="Postdoc Role", job_type_inferred="postdoc")
    _insert_job(
        conn,
        job_id="51",
        title="PhD Role",
        job_type_inferred="phd",
        cleaned_text="PhD role with postdoc collaboration topics.",
    )
    indexing.rebuild_fts(conn)

    rows = search.hybrid_search(
        conn,
        query="postdoc role",
        job_type="phd",
        active_only=False,
        open_only=False,
        limit=20,
        index_dir=MISSING_INDEX_DIR,
    )

    assert rows
    assert {row["job_type_inferred"] for row in rows} == {"phd"}


def test_reranker_reorders_synthetic_candidates(monkeypatch):
    class FakeModel:
        def encode(self, texts, **_kwargs):
            vectors = []
            for text in texts:
                lowered = str(text).lower()
                if "quantum optics" in lowered:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    monkeypatch.setattr(search, "_load_embedding_model", lambda _name: FakeModel())

    candidates = [
        {
            "meta": {
                "title": "General Lab Position",
                "organization": "Org A",
                "cleaned_text": "General science role without optics focus.",
                "job_type_inferred": "postdoc",
                "topic_domain": "natural_sciences",
            },
            "score": 0.09,
            "score_components": None,
        },
        {
            "meta": {
                "title": "Postdoctoral Position in Quantum Optics",
                "organization": "Org B",
                "cleaned_text": "Quantum optics and photonics research.",
                "job_type_inferred": "postdoc",
                "topic_domain": "natural_sciences",
            },
            "score": 0.04,
            "score_components": None,
        },
    ]

    reranked = search.rerank_candidates(
        "quantum optics postdoc",
        candidates,
        model_name="fake",
        debug=True,
    )
    assert reranked[0]["meta"]["title"] == "Postdoctoral Position in Quantum Optics"
    assert "score_components" in reranked[0]
