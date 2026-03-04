from __future__ import annotations

import json
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
    "job_type_inferred",
    "job_type_score",
    "topic_domain",
    "topic_confidence",
    "topic_scores_json",
    "topic_model",
    "topic_updated_at",
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
    job_type_inferred TEXT,
    job_type_score INTEGER,
    topic_domain TEXT,
    topic_confidence REAL,
    topic_scores_json TEXT,
    topic_model TEXT,
    topic_updated_at TEXT,
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


def _jobs_table_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(jobs)").fetchall()
    return {row["name"] for row in rows}


def _safe_add_column(conn: sqlite3.Connection, table: str, column_name: str, column_type: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")
    except sqlite3.OperationalError as exc:
        # Handle concurrent startup races (duplicate column) gracefully.
        if "duplicate column name" not in str(exc).lower():
            raise


def _ensure_jobs_schema_migrations(conn: sqlite3.Connection) -> None:
    columns = _jobs_table_columns(conn)
    if "job_type_inferred" not in columns:
        _safe_add_column(conn, "jobs", "job_type_inferred", "TEXT")
    if "job_type_score" not in columns:
        _safe_add_column(conn, "jobs", "job_type_score", "INTEGER")
    if "topic_domain" not in columns:
        _safe_add_column(conn, "jobs", "topic_domain", "TEXT")
    if "topic_confidence" not in columns:
        _safe_add_column(conn, "jobs", "topic_confidence", "REAL")
    if "topic_scores_json" not in columns:
        _safe_add_column(conn, "jobs", "topic_scores_json", "TEXT")
    if "topic_model" not in columns:
        _safe_add_column(conn, "jobs", "topic_model", "TEXT")
    if "topic_updated_at" not in columns:
        _safe_add_column(conn, "jobs", "topic_updated_at", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type_inferred ON jobs(job_type_inferred)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_topic_domain ON jobs(topic_domain)")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    _ensure_jobs_schema_migrations(conn)
    # Backfill rows affected by older logic that persisted 304 on previously fetched jobs.
    # Guarded so it only runs once, not on every call (important for the web server).
    if get_state(conn, "migration:304_backfill") is None:
        conn.execute(
            """
            UPDATE jobs
            SET http_status = 200
            WHERE http_status = 304
              AND cleaned_text IS NOT NULL
              AND TRIM(cleaned_text) != ''
            """
        )
        set_state(conn, "migration:304_backfill", "done")
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


def get_job_detail(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM jobs
        WHERE job_id = ?
          AND http_status = 200
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        """,
        (job_id,),
    ).fetchone()


def parse_sections_json(value: sqlite3.Row | dict[str, Any] | str | None) -> dict[str, str]:
    raw = value
    if isinstance(value, sqlite3.Row):
        raw = value["sections_json"]
    elif isinstance(value, dict):
        raw = value.get("sections_json")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for key, text in payload.items():
        if not isinstance(key, str):
            continue
        out[key] = str(text or "").strip()
    return out


def jobs_for_reclassification(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    sql = """
        SELECT job_id, title, researcher_profile, cleaned_text
        FROM jobs
        WHERE http_status = 200
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        ORDER BY job_id
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    return conn.execute(sql, params).fetchall()


def jobs_for_topic_classification(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    only_missing: bool = False,
    since: str | None = None,
) -> list[sqlite3.Row]:
    clauses = [
        "http_status = 200",
        "cleaned_text IS NOT NULL",
        "TRIM(cleaned_text) != ''",
    ]
    params: list[Any] = []
    if only_missing:
        clauses.append("(topic_domain IS NULL OR TRIM(topic_domain) = '')")
    if since:
        clauses.append("last_seen_at >= ?")
        params.append(since)

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT job_id, title, cleaned_text, sections_json
        FROM jobs
        WHERE {where_sql}
        ORDER BY job_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return conn.execute(sql, tuple(params)).fetchall()


def update_job_type_classification(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    job_type_inferred: str,
    job_type_score: int,
) -> None:
    conn.execute(
        """
        UPDATE jobs
        SET job_type_inferred = ?, job_type_score = ?
        WHERE job_id = ?
        """,
        (job_type_inferred, int(job_type_score), job_id),
    )


def update_job_topic_classification(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    topic_domain: str,
    topic_confidence: float,
    topic_scores_json: str,
    topic_model: str,
    topic_updated_at: str,
) -> None:
    conn.execute(
        """
        UPDATE jobs
        SET topic_domain = ?,
            topic_confidence = ?,
            topic_scores_json = ?,
            topic_model = ?,
            topic_updated_at = ?
        WHERE job_id = ?
        """,
        (
            topic_domain,
            float(topic_confidence),
            topic_scores_json,
            topic_model,
            topic_updated_at,
            job_id,
        ),
    )


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
