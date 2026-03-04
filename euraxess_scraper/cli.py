from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import config, db, discovery, export as export_mod, indexing, search as search_mod
from .fetch import Fetcher, GlobalHaltError
from .parse_job import parse_job_detail
from .utils import now_utc_iso, robots_diagnostic

app = typer.Typer(help="EURAXESS production crawler and hybrid-search CLI")
console = Console()
LOGGER = logging.getLogger("euraxess_scraper")


def _open_db():
    config.ensure_data_dirs()
    conn = db.get_connection(config.DB_PATH)
    db.init_db(conn)
    return conn


def _warn_user_agent(logger: logging.Logger) -> None:
    if config.user_agent_has_placeholders():
        logger.warning(
            "USER_AGENT still contains placeholders. Update config.USER_AGENT before heavy crawling."
        )


async def _process_pending_jobs(
    conn,
    fetcher: Fetcher,
    *,
    logger: logging.Logger,
    limit: int | None,
    update_mode: bool,
    concurrency: int,
) -> dict[str, Any]:
    rows = db.get_pending_jobs(conn, limit=limit)
    if not rows:
        return {
            "queued": 0,
            "fetched": 0,
            "not_modified": 0,
            "unchanged_hash": 0,
            "failed": 0,
            "retry_pending": 0,
            "halted": False,
            "halt_reason": None,
        }

    queue: asyncio.Queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(dict(row))

    counters: dict[str, Any] = {
        "queued": len(rows),
        "fetched": 0,
        "not_modified": 0,
        "unchanged_hash": 0,
        "failed": 0,
        "retry_pending": 0,
        "halted": False,
        "halt_reason": None,
    }

    db_lock = asyncio.Lock()
    halt_event = asyncio.Event()

    async def worker(worker_id: int) -> None:
        while True:
            if halt_event.is_set():
                return
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            job_id = row["job_id"]
            url = row["url"]
            try:
                result = await fetcher.get(url, etag=row.get("etag"), last_modified=row.get("last_modified"))
            except GlobalHaltError as exc:
                counters["halted"] = True
                counters["halt_reason"] = str(exc)
                halt_event.set()
                queue.task_done()
                return

            now_iso = now_utc_iso()

            if result.status == 304:
                async with db_lock:
                    db.touch_job_not_modified(conn, job_id, now_iso, result.etag, result.last_modified)
                    db.mark_queue_done(conn, job_id)
                counters["not_modified"] += 1
                logger.info("Job %s not modified (304)", job_id)
                queue.task_done()
                continue

            if result.status == 200 and result.text:
                try:
                    parsed = parse_job_detail(result.text, result.final_url or url)
                except Exception as exc:  # pragma: no cover - parser hardening path
                    error = f"parse error: {exc}"
                    async with db_lock:
                        attempts = db.bump_attempt(conn, job_id, now_iso, error)
                        if attempts >= config.MAX_RETRIES:
                            db.mark_queue_failed(conn, job_id, error)
                            counters["failed"] += 1
                        else:
                            counters["retry_pending"] += 1
                    logger.warning("Job %s parse failed (attempt %s): %s", job_id, attempts, error)
                    queue.task_done()
                    continue

                async with db_lock:
                    previous = db.get_job_row(conn, job_id)
                    old_hash = previous["content_hash"] if previous else None

                    parsed["job_id"] = job_id
                    parsed["url"] = url
                    parsed["etag"] = result.etag or (previous["etag"] if previous else None)
                    parsed["last_modified"] = result.last_modified or (
                        previous["last_modified"] if previous else None
                    )
                    parsed["first_seen_at"] = (
                        previous["first_seen_at"] if previous else now_iso
                    )
                    parsed["last_seen_at"] = now_iso
                    parsed["delisted_at"] = None
                    parsed["fetched_at"] = now_iso
                    parsed["http_status"] = 200
                    parsed["error"] = None

                    if update_mode and old_hash and parsed.get("content_hash") == old_hash:
                        conn.execute(
                            """
                            UPDATE jobs
                            SET last_seen_at = ?, fetched_at = ?, http_status = 200,
                                error = NULL, delisted_at = NULL,
                                etag = COALESCE(?, etag),
                                last_modified = COALESCE(?, last_modified)
                            WHERE job_id = ?
                            """,
                            (now_iso, now_iso, result.etag, result.last_modified, job_id),
                        )
                        conn.commit()
                        db.mark_queue_done(conn, job_id)
                        counters["unchanged_hash"] += 1
                        logger.info("Job %s unchanged (content hash)", job_id)
                    else:
                        db.upsert_job_detail(conn, parsed)
                        db.mark_queue_done(conn, job_id)
                        counters["fetched"] += 1
                        logger.info("Job %s fetched and stored", job_id)

                queue.task_done()
                continue

            error = result.error or f"http_status={result.status}"
            async with db_lock:
                attempts = db.bump_attempt(conn, job_id, now_iso, error)
                if attempts >= config.MAX_RETRIES:
                    db.mark_queue_failed(conn, job_id, error)
                    counters["failed"] += 1
                    logger.warning("Job %s failed permanently after %s attempts", job_id, attempts)
                else:
                    counters["retry_pending"] += 1
                    logger.warning("Job %s failed attempt %s/%s", job_id, attempts, config.MAX_RETRIES)
            queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(max(1, concurrency))]
    await asyncio.gather(*workers)
    return counters


async def _crawl_or_update(
    *,
    update_mode: bool,
    limit: int | None,
    dry_run: bool,
    rps: float,
    concurrency: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    conn = _open_db()
    _warn_user_agent(logger)

    run_key = "update" if update_mode else "crawl"
    db.set_state(conn, f"{run_key}:last_start", now_utc_iso())

    robots = robots_diagnostic(config.SEARCH_URL, config.USER_AGENT)
    if robots["error"]:
        logger.warning("robots.txt check failed: %s", robots["error"])
    elif not robots["allowed"]:
        logger.warning("robots.txt disallows target path; continuing as requested")
    else:
        logger.info("robots.txt allows target path")

    active_before = db.get_active_job_ids(conn) if update_mode else set()

    async with Fetcher(rps=rps, user_agent=config.USER_AGENT) as fetcher:
        discovered_ids = await discovery.discover_jobs(
            conn,
            fetcher,
            logger=logger,
            requeue_existing=update_mode,
            max_jobs=(limit if (limit is not None and not update_mode) else None),
        )

        if dry_run:
            db.set_state(conn, f"{run_key}:last_end", now_utc_iso())
            return {
                "discovered": len(discovered_ids),
                "processed": 0,
                "dry_run": True,
                "halted": False,
                "halt_reason": None,
            }

        processing_result = await _process_pending_jobs(
            conn,
            fetcher,
            logger=logger,
            limit=limit,
            update_mode=update_mode,
            concurrency=concurrency,
        )

    delisted_count = 0
    if update_mode:
        missing = active_before - discovered_ids
        delisted_count = db.mark_delisted(conn, missing, now_utc_iso())
        logger.info("Marked %s jobs as delisted", delisted_count)

    db.set_state(conn, f"{run_key}:last_end", now_utc_iso())
    return {
        "discovered": len(discovered_ids),
        "dry_run": False,
        "delisted": delisted_count,
        **processing_result,
    }


@app.command()
def crawl(
    limit: int | None = typer.Option(None, "--limit", help="Max jobs to fetch details for"),
    rps: float = typer.Option(config.DEFAULT_RPS, "--rps", help="Requests per second"),
    concurrency: int = typer.Option(
        config.DEFAULT_CONCURRENCY,
        "--concurrency",
        help="Concurrent workers",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only discover listings, do not fetch detail"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    result = asyncio.run(
        _crawl_or_update(
            update_mode=False,
            limit=limit,
            dry_run=dry_run,
            rps=rps,
            concurrency=concurrency,
            logger=LOGGER,
        )
    )
    console.print(result)
    if result.get("halted"):
        raise typer.Exit(code=2)


@app.command()
def update(
    rps: float = typer.Option(config.DEFAULT_RPS, "--rps", help="Requests per second"),
    concurrency: int = typer.Option(
        config.DEFAULT_CONCURRENCY,
        "--concurrency",
        help="Concurrent workers",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    result = asyncio.run(
        _crawl_or_update(
            update_mode=True,
            limit=None,
            dry_run=False,
            rps=rps,
            concurrency=concurrency,
            logger=LOGGER,
        )
    )
    console.print(result)
    if result.get("halted"):
        raise typer.Exit(code=2)


@app.command("export")
def export_command(
    format: str = typer.Option(..., "--format", help="jsonl or parquet"),
    output: Path | None = typer.Option(None, "--output", help="Output file path"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    conn = _open_db()

    fmt = format.lower()
    if fmt not in {"jsonl", "parquet"}:
        raise typer.BadParameter("--format must be one of: jsonl, parquet")

    if output is None:
        output = config.EXPORT_DIR / ("jobs.jsonl" if fmt == "jsonl" else "jobs.parquet")

    if fmt == "jsonl":
        count = export_mod.export_jsonl(conn, output)
    else:
        count = export_mod.export_parquet(conn, output)

    console.print({"format": fmt, "output": str(output), "rows": count})


@app.command("build-index")
def build_index(
    model: str = typer.Option(config.EMBEDDING_MODEL, "--model", help="SentenceTransformer model"),
    batch_size: int = typer.Option(64, "--batch-size", help="Embedding batch size"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    conn = _open_db()
    try:
        info = indexing.build_indexes(conn, model_name=model, batch_size=batch_size)
    except ImportError as exc:
        raise typer.BadParameter(
            "Missing indexing dependencies. Install project dependencies first."
        ) from exc
    console.print(info)


@app.command()
def search(
    query: str = typer.Option(..., "--query", help="Search query"),
    top_k: int = typer.Option(10, "--top-k", help="Top K results"),
    country: str | None = typer.Option(None, "--country", help="Country filter"),
    rrf_k: int = typer.Option(config.RRF_K, "--rrf-k", help="RRF constant"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    conn = _open_db()
    try:
        rows = search_mod.hybrid_search(
            conn,
            query=query,
            top_k=top_k,
            country=country,
            rrf_k=rrf_k,
        )
    except ImportError as exc:
        raise typer.BadParameter(
            "Missing vector-search dependencies. Install project dependencies first."
        ) from exc

    table = Table(title=f"Hybrid Search Results ({len(rows)})")
    table.add_column("Score", justify="right")
    table.add_column("Title")
    table.add_column("Org")
    table.add_column("Country")
    table.add_column("Deadline")
    table.add_column("URL")
    for row in rows:
        table.add_row(
            f"{row['rrf_score']:.5f}",
            str(row.get("title") or ""),
            str(row.get("organization") or ""),
            str(row.get("country") or ""),
            str(row.get("deadline") or ""),
            str(row.get("url") or ""),
        )
    console.print(table)


@app.command()
def stats(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs")) -> None:
    config.setup_logging(verbose)
    conn = _open_db()
    snapshot = db.stats_snapshot(conn)
    idx = indexing.index_status(config.INDEX_DIR)

    table = Table(title="EURAXESS Scraper Stats")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total jobs", str(snapshot["total_jobs"]))
    table.add_row("Queue pending", str(snapshot["queue_counts"].get("pending", 0)))
    table.add_row("Queue done", str(snapshot["queue_counts"].get("done", 0)))
    table.add_row("Queue failed", str(snapshot["queue_counts"].get("failed", 0)))
    table.add_row("Delisted", str(snapshot["delisted_count"]))
    table.add_row("Last crawl", str(snapshot["last_crawl_end"]))
    table.add_row("Last update", str(snapshot["last_update_end"]))
    table.add_row("FAISS exists", str(idx["faiss_exists"]))
    table.add_row("Mapping exists", str(idx["mapping_exists"]))
    table.add_row("Indexed vectors", str(idx["vectors"]))
    console.print(table)


if __name__ == "__main__":
    app()
