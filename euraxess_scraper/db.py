from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import DB_PATH

JOB_COLUMNS = [
    "job_id",
    "url",
    "title",
    "organization",
    "country",
    "city",
    "posted_date",
    "deadline",
    "researcher_profile",
    "position_type",
    "contract_type",
    "hours",
    "salary",
    "cleaned_text",
    "sections_json",
    "content_hash",
    "etag",
    "last_modified",
    "first_seen_at",
    "last_seen_at",
    "delisted_at",
    "fetched_at",
    "http_status",
    "error",
]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    url          TEXT UNIQUE NOT NULL,
    title        TEXT,
    organization TEXT,
    country      TEXT,
    city         TEXT,
    posted_date  TEXT,
    deadline     TEXT,
    researcher_profile TEXT,
    position_type TEXT,
    contract_type TEXT,
    hours        TEXT,
    salary       TEXT,
    cleaned_text TEXT,
    sections_json TEXT,
    content_hash TEXT,
    etag         TEXT,
    last_modified TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    delisted_at   TEXT,
    fetched_at   TEXT,
    http_status  INTEGER,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS queue (
    job_id         TEXT PRIMARY KEY,
    status         TEXT NOT NULL DEFAULT 'pending',
    attempts       INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT
);

CREATE TABLE IF NOT EXISTS crawl_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_last_seen_at ON jobs(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_jobs_delisted_at ON jobs(delisted_at);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_attempts ON queue(attempts);
"""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    # Backfill rows affected by older logic that persisted 304 on previously fetched jobs.
    conn.execute(
        """
        UPDATE jobs
        SET http_status = 200
        WHERE http_status = 304
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        """
    )
    conn.commit()


def upsert_job_stub(conn: sqlite3.Connection, job_id: str, url: str, now_iso: str) -> bool:
    existing = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.execute(
        """
        INSERT INTO jobs (job_id, url, position_type, first_seen_at, last_seen_at)
        VALUES (?, ?, 'job_offer', ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
          url = excluded.url,
          last_seen_at = excluded.last_seen_at,
          delisted_at = NULL
        """,
        (job_id, url, now_iso, now_iso),
    )
    conn.commit()
    return existing is None


def enqueue_pending(conn: sqlite3.Connection, job_id: str, force: bool = False) -> None:
    if force:
        conn.execute(
            """
            INSERT INTO queue(job_id, status, attempts, last_attempt_at)
            VALUES (?, 'pending', 0, NULL)
            ON CONFLICT(job_id) DO UPDATE SET
              status='pending',
              attempts=0,
              last_attempt_at=NULL
            """,
            (job_id,),
        )
    else:
        conn.execute(
            """
            INSERT INTO queue(job_id, status)
            VALUES (?, 'pending')
            ON CONFLICT(job_id) DO NOTHING
            """,
            (job_id,),
        )
    conn.commit()


def get_pending_jobs(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    sql = """
        SELECT q.job_id, q.status, q.attempts, q.last_attempt_at,
               j.url, j.etag, j.last_modified, j.content_hash
        FROM queue q
        JOIN jobs j ON j.job_id = q.job_id
        WHERE q.status = 'pending'
        ORDER BY COALESCE(q.last_attempt_at, '') ASC, q.job_id ASC
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()
    return rows


def mark_queue_done(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE queue SET status='done', attempts=0 WHERE job_id = ?",
        (job_id,),
    )
    conn.commit()


def mark_queue_failed(conn: sqlite3.Connection, job_id: str, error: str) -> None:
    conn.execute("UPDATE queue SET status='failed' WHERE job_id = ?", (job_id,))
    conn.execute("UPDATE jobs SET error = ? WHERE job_id = ?", (error, job_id))
    conn.commit()


def bump_attempt(conn: sqlite3.Connection, job_id: str, now_iso: str, error: str) -> int:
    conn.execute(
        """
        UPDATE queue
        SET attempts = attempts + 1,
            last_attempt_at = ?
        WHERE job_id = ?
        """,
        (now_iso, job_id),
    )
    conn.execute("UPDATE jobs SET error = ? WHERE job_id = ?", (error, job_id))
    attempts_row = conn.execute("SELECT attempts FROM queue WHERE job_id = ?", (job_id,)).fetchone()
    conn.commit()
    return int(attempts_row["attempts"]) if attempts_row else 0


def upsert_job_detail(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    columns = ", ".join(JOB_COLUMNS)
    placeholders = ", ".join(["?"] * len(JOB_COLUMNS))
    updates = ", ".join([f"{col}=excluded.{col}" for col in JOB_COLUMNS if col != "job_id"])

    values = [record.get(col) for col in JOB_COLUMNS]
    conn.execute(
        f"""
        INSERT INTO jobs ({columns})
        VALUES ({placeholders})
        ON CONFLICT(job_id) DO UPDATE SET {updates}
        """,
        values,
    )
    conn.commit()


def touch_job_not_modified(
    conn: sqlite3.Connection,
    job_id: str,
    now_iso: str,
    etag: str | None,
    last_modified: str | None,
) -> None:
    conn.execute(
        """
        UPDATE jobs
        SET fetched_at = ?,
            last_seen_at = ?,
            error = NULL,
            delisted_at = NULL,
            etag = COALESCE(?, etag),
            last_modified = COALESCE(?, last_modified)
        WHERE job_id = ?
        """,
        (now_iso, now_iso, etag, last_modified, job_id),
    )
    conn.commit()


def touch_job_seen(conn: sqlite3.Connection, job_id: str, now_iso: str) -> None:
    conn.execute(
        "UPDATE jobs SET last_seen_at = ?, delisted_at = NULL WHERE job_id = ?",
        (now_iso, job_id),
    )
    conn.commit()


def mark_delisted(conn: sqlite3.Connection, missing_job_ids: set[str], now_iso: str) -> int:
    if not missing_job_ids:
        return 0
    placeholders = ",".join(["?"] * len(missing_job_ids))
    sql = f"UPDATE jobs SET delisted_at = ? WHERE job_id IN ({placeholders}) AND delisted_at IS NULL"
    params: list[Any] = [now_iso]
    params.extend(sorted(missing_job_ids))
    cur = conn.execute(sql, tuple(params))
    conn.commit()
    return cur.rowcount


def clear_delisted(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute("UPDATE jobs SET delisted_at = NULL WHERE job_id = ?", (job_id,))
    conn.commit()


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM crawl_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO crawl_state(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


def get_active_job_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT job_id FROM jobs WHERE delisted_at IS NULL").fetchall()
    return {row["job_id"] for row in rows}


def get_job_row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()


def jobs_for_export(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT * FROM jobs
        WHERE http_status = 200
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        ORDER BY job_id
        """
    ).fetchall()
    return rows


def stats_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    total_jobs = conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
    queue_counts = {
        row["status"]: row["c"]
        for row in conn.execute("SELECT status, COUNT(*) AS c FROM queue GROUP BY status").fetchall()
    }
    delisted_count = conn.execute(
        "SELECT COUNT(*) AS c FROM jobs WHERE delisted_at IS NOT NULL"
    ).fetchone()["c"]
    return {
        "total_jobs": total_jobs,
        "queue_counts": queue_counts,
        "delisted_count": delisted_count,
        "last_crawl_end": get_state(conn, "crawl:last_end"),
        "last_update_end": get_state(conn, "update:last_end"),
    }
