from __future__ import annotations

import sqlite3

from euraxess_scraper import db, search


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
    )

    assert rows
    assert {row["job_type_inferred"] for row in rows} == {"phd"}


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

    rows = search.hybrid_search(conn, query="", limit=50)
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
    )
    job_types = {row["job_type_inferred"] for row in rows}

    assert {"postdoc", "phd", "professor"}.issubset(job_types)

