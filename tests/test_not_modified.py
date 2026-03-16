from __future__ import annotations

import sqlite3

from euraxess_scraper import db


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    return conn


def test_touch_job_not_modified_keeps_existing_http_status():
    conn = _conn()
    first_seen = "2026-03-04T00:00:00Z"
    refreshed = "2026-03-04T01:00:00Z"

    record = {column: None for column in db.JOB_COLUMNS}
    record.update(
        {
            "job_id": "123",
            "url": "https://euraxess.ec.europa.eu/jobs/123",
            "title": "Sample role",
            "cleaned_text": "sample cleaned text",
            "content_hash": "abc123",
            "first_seen_at": first_seen,
            "last_seen_at": first_seen,
            "fetched_at": first_seen,
            "http_status": 200,
        }
    )
    db.upsert_job_detail(conn, record)
    db.touch_job_not_modified(conn, "123", refreshed, etag="etag-1", last_modified="lm-1")

    row = db.get_job_row(conn, "123")
    assert row is not None
    assert row["http_status"] == 200
    assert row["last_seen_at"] == refreshed
    assert row["etag"] == "etag-1"
    assert row["last_modified"] == "lm-1"
