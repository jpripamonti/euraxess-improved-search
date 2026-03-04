from __future__ import annotations

import sqlite3

from euraxess_scraper import db


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    return conn


def test_deduplicate_same_job_id_inserted_twice():
    conn = _conn()
    now = "2026-03-03T12:00:00Z"

    db.upsert_job_stub(conn, "123", "https://euraxess.ec.europa.eu/jobs/123", now)
    db.enqueue_pending(conn, "123")

    db.upsert_job_stub(conn, "123", "https://euraxess.ec.europa.eu/jobs/123", now)
    db.enqueue_pending(conn, "123")

    jobs_count = conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
    queue_count = conn.execute("SELECT COUNT(*) AS c FROM queue").fetchone()["c"]

    assert jobs_count == 1
    assert queue_count == 1
