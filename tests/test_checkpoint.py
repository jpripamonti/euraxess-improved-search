from __future__ import annotations

import sqlite3

from euraxess_scraper import db


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    return conn


def test_checkpoint_and_resume_flow_in_memory_db():
    conn = _conn()
    now = "2026-03-03T12:00:00Z"

    db.upsert_job_stub(conn, "100", "https://euraxess.ec.europa.eu/jobs/100", now)
    db.upsert_job_stub(conn, "101", "https://euraxess.ec.europa.eu/jobs/101", now)
    db.enqueue_pending(conn, "100")
    db.enqueue_pending(conn, "101")

    db.set_state(conn, "discovery:last_page", "12")
    assert db.get_state(conn, "discovery:last_page") == "12"

    db.mark_queue_done(conn, "100")

    pending = db.get_pending_jobs(conn)
    pending_ids = [row["job_id"] for row in pending]

    assert pending_ids == ["101"]
