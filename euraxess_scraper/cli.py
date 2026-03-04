from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from . import config, db, discovery, export as export_mod, indexing, search as search_mod, taxonomy, topics
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


def _parse_gold_cases(gold_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(gold_path.read_text(encoding="utf-8")) or {}
    if isinstance(payload, list):
        cases = payload
    else:
        cases = payload.get("queries") or []
    out: list[dict[str, Any]] = []
    for item in cases:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        raw_rel = item.get("relevant") or {}
        relevance: dict[str, float] = {}
        if isinstance(raw_rel, dict):
            for job_id, score in raw_rel.items():
                try:
                    relevance[str(job_id)] = float(score)
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw_rel, list):
            for job_id in raw_rel:
                relevance[str(job_id)] = 1.0
        if not relevance:
            continue
        out.append(
            {
                "query": query,
                "relevant": relevance,
                "job_type": item.get("job_type"),
                "topic": item.get("topic"),
                "country": item.get("country"),
                "active_only": bool(item.get("active_only", True)),
                "open_only": bool(item.get("open_only", True)),
            }
        )
    return out


def _dcg(scores: list[float]) -> float:
    total = 0.0
    for idx, rel in enumerate(scores, start=1):
        total += (2.0**float(rel) - 1.0) / math.log2(idx + 1.0)
    return total


def _evaluate_mode(
    conn,
    cases: list[dict[str, Any]],
    *,
    mode: str,
    k: int,
) -> dict[str, float]:
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []

    enable_rerank = mode != "hybrid_no_rerank"
    for case in cases:
        relevant = case["relevant"]
        rows = search_mod.hybrid_search(
            conn,
            query=case["query"],
            limit=k,
            country=case.get("country"),
            job_type=case.get("job_type"),
            topic_domain=case.get("topic"),
            active_only=case.get("active_only", True),
            open_only=case.get("open_only", True),
            enable_rerank=enable_rerank,
        )
        predicted_ids = [str(row.get("job_id")) for row in rows]
        gains = [float(relevant.get(job_id, 0.0)) for job_id in predicted_ids[:k]]
        ideal = sorted([float(v) for v in relevant.values()], reverse=True)[:k]
        dcg = _dcg(gains)
        idcg = _dcg(ideal)
        ndcgs.append(dcg / idcg if idcg > 0.0 else 0.0)

        rr = 0.0
        for rank, job_id in enumerate(predicted_ids[:k], start=1):
            if float(relevant.get(job_id, 0.0)) > 0.0:
                rr = 1.0 / float(rank)
                break
        mrrs.append(rr)

        relevant_ids = {str(job_id) for job_id in relevant.keys()}
        hit_count = len([job_id for job_id in predicted_ids[:k] if job_id in relevant_ids])
        recalls.append(float(hit_count) / float(len(relevant_ids)) if relevant_ids else 0.0)

    denom = max(1, len(cases))
    return {
        "queries": float(len(cases)),
        "ndcg@k": float(sum(ndcgs) / denom),
        "mrr@k": float(sum(mrrs) / denom),
        "recall@k": float(sum(recalls) / denom),
    }


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
    max_pages: int | None,
    mark_delisted: bool,
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
        discovery_result = await discovery.discover_jobs(
            conn,
            fetcher,
            logger=logger,
            requeue_existing=update_mode,
            max_pages=max_pages,
            max_jobs=(limit if (limit is not None and not update_mode) else None),
        )

        if dry_run:
            db.set_state(conn, f"{run_key}:last_end", now_utc_iso())
            return {
                "discovered": len(discovery_result.discovered_ids),
                "processed": 0,
                "dry_run": True,
                "discovery_completed": discovery_result.completed,
                "discovery_stop_reason": discovery_result.stop_reason,
                "discovery_last_page": discovery_result.last_processed_page,
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
    discovered_ids = discovery_result.discovered_ids
    full_scan = (
        discovery_result.completed
        and max_pages is None
        and discovery_result.stop_reason in {"no_links", "no_new_ids"}
    )
    if update_mode:
        if mark_delisted and full_scan:
            missing = active_before - discovered_ids
            delisted_count = db.mark_delisted(conn, missing, now_utc_iso())
            logger.info("Marked %s jobs as delisted", delisted_count)
        elif mark_delisted:
            logger.warning(
                "Skipped delisted marking because discovery was not a complete scan "
                "(completed=%s, stop_reason=%s, max_pages=%s)",
                discovery_result.completed,
                discovery_result.stop_reason,
                max_pages,
            )

    db.set_state(conn, f"{run_key}:last_end", now_utc_iso())
    return {
        "discovered": len(discovered_ids),
        "dry_run": False,
        "delisted": delisted_count,
        "discovery_completed": discovery_result.completed,
        "discovery_stop_reason": discovery_result.stop_reason,
        "discovery_last_page": discovery_result.last_processed_page,
        "discovery_full_scan": full_scan,
        **processing_result,
    }


@app.command()
def crawl(
    limit: int | None = typer.Option(None, "--limit", help="Max jobs to fetch details for"),
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        min=1,
        help="Limit discovery to the first N listing pages",
    ),
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
            max_pages=max_pages,
            mark_delisted=False,
            logger=LOGGER,
        )
    )
    console.print(result)
    if result.get("halted"):
        raise typer.Exit(code=2)


@app.command()
def update(
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        min=1,
        help="Limit discovery to the first N listing pages (for faster incremental runs)",
    ),
    delist: bool = typer.Option(
        True,
        "--delist/--no-delist",
        help="Mark jobs as delisted only after a complete discovery scan",
    ),
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
            max_pages=max_pages,
            mark_delisted=delist,
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
def reclassify(
    limit: int | None = typer.Option(None, "--limit", min=1, help="Only reclassify the first N jobs"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    conn = _open_db()

    rows = db.jobs_for_reclassification(conn, limit=limit)
    updated = 0
    for row in rows:
        classification = taxonomy.classify_job_type(
            title=row["title"],
            researcher_profile=row["researcher_profile"],
            cleaned_text=row["cleaned_text"],
        )
        db.update_job_type_classification(
            conn,
            job_id=row["job_id"],
            job_type_inferred=classification.job_type,
            job_type_score=classification.score,
        )
        updated += 1
    conn.commit()

    counts = conn.execute(
        """
        SELECT COALESCE(job_type_inferred, 'unknown') AS job_type, COUNT(*) AS c
        FROM jobs
        WHERE http_status = 200
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        GROUP BY COALESCE(job_type_inferred, 'unknown')
        ORDER BY c DESC
        """
    ).fetchall()
    breakdown = {row["job_type"]: int(row["c"]) for row in counts}
    console.print({"updated": updated, "breakdown": breakdown})


@app.command("classify-topics")
def classify_topics(
    limit: int | None = typer.Option(None, "--limit", min=1, help="Only classify the first N jobs"),
    only_missing: bool = typer.Option(
        False,
        "--only-missing/--include-classified",
        help="Classify only jobs without a stored topic",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Only classify jobs with last_seen_at >= this ISO timestamp",
    ),
    batch_size: int = typer.Option(64, "--batch-size", min=1, help="Batch size for embedding inference"),
    model: str = typer.Option(topics.DEFAULT_TOPIC_MODEL, "--model", help="Embedding model for topic tagging"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    conn = _open_db()
    rows = db.jobs_for_topic_classification(
        conn,
        limit=limit,
        only_missing=only_missing,
        since=since,
    )
    if not rows:
        console.print({"updated": 0, "breakdown": {}})
        return

    updated = 0
    breakdown: dict[str, int] = {}
    topic_updated_at = now_utc_iso()
    for start in range(0, len(rows), max(1, int(batch_size))):
        row_chunk = rows[start : start + max(1, int(batch_size))]
        item_chunk = [
            {
                "title": row["title"],
                "cleaned_text": row["cleaned_text"],
                "sections": db.parse_sections_json(row),
            }
            for row in row_chunk
        ]
        classified = topics.classify_topics_batch(
            item_chunk,
            model_name=model,
            batch_size=batch_size,
        )
        for row, result in zip(row_chunk, classified):
            db.update_job_topic_classification(
                conn,
                job_id=str(row["job_id"]),
                topic_domain=result.topic_domain,
                topic_confidence=result.confidence,
                topic_scores_json=json.dumps(result.scores, ensure_ascii=False, sort_keys=True),
                topic_model=model,
                topic_updated_at=topic_updated_at,
            )
            updated += 1
            breakdown[result.topic_domain] = breakdown.get(result.topic_domain, 0) + 1
        conn.commit()

    counts = conn.execute(
        """
        SELECT COALESCE(topic_domain, 'other') AS topic_domain, COUNT(*) AS c
        FROM jobs
        WHERE http_status = 200
          AND cleaned_text IS NOT NULL
          AND TRIM(cleaned_text) != ''
        GROUP BY COALESCE(topic_domain, 'other')
        ORDER BY c DESC
        """
    ).fetchall()
    global_breakdown = {row["topic_domain"]: int(row["c"]) for row in counts}
    console.print(
        {
            "updated": updated,
            "batch_breakdown": breakdown,
            "global_breakdown": global_breakdown,
        }
    )


@app.command()
def search(
    query: str = typer.Option(..., "--query", help="Search query"),
    top_k: int = typer.Option(10, "--top-k", help="Top K results"),
    country: str | None = typer.Option(None, "--country", help="Country filter"),
    job_type: str | None = typer.Option(
        None,
        "--job-type",
        help="Filter by type: all, postdoc, phd, professor",
    ),
    topic: str | None = typer.Option(
        None,
        "--topic",
        help="Filter by topic: all, natural_sciences, engineering_technology, medical_health, agricultural_veterinary, social_sciences, humanities_arts, other",
    ),
    active_only: bool = typer.Option(
        True,
        "--active-only/--include-delisted",
        help="Exclude delisted positions by default",
    ),
    open_only: bool = typer.Option(
        True,
        "--open-only/--include-closed",
        help="Exclude positions with past deadlines by default",
    ),
    rrf_k: int = typer.Option(config.RRF_K, "--rrf-k", help="RRF constant"),
    model: str = typer.Option(config.EMBEDDING_MODEL, "--model", help="Embedding model for semantic search"),
    keyword_weight: float = typer.Option(1.0, "--keyword-weight", help="Weight for keyword (FTS) ranking"),
    vector_weight: float = typer.Option(2.0, "--vector-weight", help="Weight for semantic vector ranking"),
    semantic_only: bool = typer.Option(
        False,
        "--semantic-only",
        help="Use semantic vector search only (no FTS keyword ranking)",
    ),
    no_rerank: bool = typer.Option(
        False,
        "--no-rerank",
        help="Disable reranking (baseline hybrid order only)",
    ),
    debug: bool = typer.Option(False, "--debug", help="Include score component diagnostics"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    conn = _open_db()
    try:
        rows = search_mod.hybrid_search(
            conn,
            query=query,
            limit=top_k,
            country=country,
            job_type=job_type,
            topic_domain=topic,
            active_only=active_only,
            open_only=open_only,
            rrf_k=rrf_k,
            model_name=model,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            semantic_only=semantic_only,
            enable_rerank=(not no_rerank),
            debug=debug,
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
    table.add_column("Type")
    table.add_column("Topic")
    table.add_column("Deadline")
    table.add_column("URL")
    for row in rows:
        score = row.get("rrf_score")
        table.add_row(
            f"{float(score):.5f}" if score is not None else "-",
            str(row.get("title") or ""),
            str(row.get("organization") or ""),
            str(row.get("country") or ""),
            str(row.get("job_type_inferred") or "unknown"),
            str(row.get("topic_domain") or topics.TOPIC_OTHER),
            str(row.get("deadline") or ""),
            str(row.get("url") or ""),
        )
    console.print(table)


@app.command("eval-search")
def eval_search(
    gold: Path = typer.Option(..., "--gold", exists=True, file_okay=True, dir_okay=False, readable=True),
    k: int = typer.Option(10, "--k", min=1, max=100, help="Cutoff for ranking metrics"),
    baseline_mode: str = typer.Option(
        "hybrid_no_rerank",
        "--baseline-mode",
        help="Baseline mode: hybrid_no_rerank",
    ),
    candidate_mode: str = typer.Option(
        "hybrid_rerank",
        "--candidate-mode",
        help="Candidate mode: hybrid_rerank",
    ),
    min_ndcg_gain: float = typer.Option(
        0.05,
        "--min-ndcg-gain",
        help="Minimum nDCG@k gain required for pass",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logs"),
) -> None:
    config.setup_logging(verbose)
    valid_modes = {"hybrid_no_rerank", "hybrid_rerank"}
    if baseline_mode not in valid_modes:
        raise typer.BadParameter(f"Unsupported --baseline-mode {baseline_mode!r}")
    if candidate_mode not in valid_modes:
        raise typer.BadParameter(f"Unsupported --candidate-mode {candidate_mode!r}")

    cases = _parse_gold_cases(gold)
    if not cases:
        raise typer.BadParameter("Gold query file has no valid query cases with relevance labels.")

    conn = _open_db()
    baseline = _evaluate_mode(conn, cases, mode=baseline_mode, k=k)
    candidate = _evaluate_mode(conn, cases, mode=candidate_mode, k=k)

    ndcg_gain = candidate["ndcg@k"] - baseline["ndcg@k"]
    passed = ndcg_gain >= float(min_ndcg_gain)
    console.print(
        {
            "k": k,
            "queries": int(baseline["queries"]),
            "baseline_mode": baseline_mode,
            "candidate_mode": candidate_mode,
            "baseline": baseline,
            "candidate": candidate,
            "ndcg_gain": ndcg_gain,
            "min_ndcg_gain": min_ndcg_gain,
            "pass": passed,
        }
    )


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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface"),
    port: int = typer.Option(8000, "--port", help="TCP port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)"),
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "Missing web dependencies. Install project dependencies first."
        ) from exc

    uvicorn.run(
        "euraxess_scraper.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    app()
