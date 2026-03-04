from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

from . import config, db
from .fetch import Fetcher
from .utils import (
    canonicalize_url,
    extract_job_id,
    extract_page_numbers_from_links,
    now_utc_iso,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    discovered_ids: set[str]
    completed: bool
    stop_reason: str | None
    last_processed_page: int


def page_url(search_url: str, page_number: int) -> str:
    parsed = urlparse(search_url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page_number)]
    encoded = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=encoded))


def extract_job_links_from_html(html: str, base_url: str = config.BASE_URL) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        if not href:
            continue
        url = canonicalize_url(href, base_url=base_url)
        job_id = extract_job_id(url)
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        out.append((job_id, url))
    return out


def extract_pagination_pages_from_html(html: str) -> list[int]:
    soup = BeautifulSoup(html, "lxml")
    hrefs = [anchor.get("href", "") for anchor in soup.select(".ecl-pagination a[href]")]
    return extract_page_numbers_from_links(hrefs)


def _json_candidates(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        if "/jobs/" in value:
            urls.append(value)
        return urls
    if isinstance(value, dict):
        for item in value.values():
            urls.extend(_json_candidates(item))
        return urls
    if isinstance(value, list):
        for item in value:
            urls.extend(_json_candidates(item))
    return urls


def extract_job_links_from_json(payload: Any, base_url: str = config.BASE_URL) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for raw in _json_candidates(payload):
        url = canonicalize_url(raw, base_url=base_url)
        job_id = extract_job_id(url)
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        out.append((job_id, url))
    return out


async def probe_endpoints(fetcher: Fetcher, logger: logging.Logger | None = None) -> str:
    logger = logger or LOGGER
    candidates = [
        f"{config.BASE_URL}/api/",
        config.SEARCH_URL + "&_format=json",
        config.SEARCH_URL + "&ajax=1",
    ]
    for url in candidates:
        result = await fetcher.get(url)
        if result.status == 200 and result.text:
            try:
                payload = json.loads(result.text)
            except json.JSONDecodeError:
                logger.info("Probe %s returned non-JSON (%s); using HTML mode", url, result.status)
                continue
            links = extract_job_links_from_json(payload)
            if links:
                logger.info("Probe %s returned usable JSON; switching discovery to JSON mode", url)
                return url
        logger.info("Probe %s status=%s; using HTML mode", url, result.status)
    return ""


async def discover_jobs(
    conn,
    fetcher: Fetcher,
    *,
    logger: logging.Logger | None = None,
    requeue_existing: bool = False,
    max_pages: int | None = None,
    max_jobs: int | None = None,
    max_page_failures: int = 6,
    retry_cooldown_seconds: int = 20,
) -> DiscoveryResult:
    logger = logger or LOGGER

    json_probe_url = await probe_endpoints(fetcher, logger=logger)
    mode = "json" if json_probe_url else "html"

    discovered_ids: set[str] = set()
    page_seen_ids: set[str] = set()
    page = 0
    last_processed_page = -1
    stop_reason: str | None = None
    completed = True
    page_failures = 0

    while True:
        if max_pages is not None and page >= max_pages:
            stop_reason = "max_pages"
            break

        base_page_url = json_probe_url if mode == "json" else config.SEARCH_URL
        url = page_url(base_page_url, page)
        result = await fetcher.get(url)
        if result.status != 200 or not result.text:
            page_failures += 1
            logger.warning(
                "Discovery page %s fetch failed with status %s (attempt %s/%s)",
                page,
                result.status,
                page_failures,
                max_page_failures,
            )
            if page_failures < max_page_failures:
                cooldown = retry_cooldown_seconds * page_failures
                if result.status == 429:
                    cooldown = max(cooldown, retry_cooldown_seconds * 2)
                    logger.warning("Rate limited on discovery; sleeping %ss before retry", cooldown)
                else:
                    logger.warning("Sleeping %ss before retrying discovery page %s", cooldown, page)
                await asyncio.sleep(cooldown)
                continue
            completed = False
            stop_reason = f"page_{page}_status_{result.status}"
            break

        page_failures = 0

        if mode == "json":
            try:
                payload = json.loads(result.text)
                links = extract_job_links_from_json(payload)
            except json.JSONDecodeError:
                logger.warning("JSON mode returned invalid JSON at page %s; falling back to HTML", page)
                mode = "html"
                links = extract_job_links_from_html(result.text)
        else:
            links = extract_job_links_from_html(result.text)

        if not links:
            logger.info("Discovery page %s has no job links; stopping", page)
            stop_reason = "no_links"
            break

        new_on_page = 0
        now_iso = now_utc_iso()
        for job_id, job_url in links:
            if max_jobs is not None and len(discovered_ids) >= max_jobs:
                break
            if job_id not in page_seen_ids:
                page_seen_ids.add(job_id)
                new_on_page += 1
            discovered_ids.add(job_id)
            db.upsert_job_stub(conn, job_id, job_url, now_iso)
            db.enqueue_pending(conn, job_id, force=requeue_existing)

        pages = extract_pagination_pages_from_html(result.text) if mode == "html" else []
        logger.info(
            "Discovery page %s fetched: %s links, %s new this run, mode=%s pagination=%s",
            page,
            len(links),
            new_on_page,
            mode,
            (max(pages) + 1) if pages else "unknown",
        )
        last_processed_page = page

        if new_on_page == 0:
            logger.info("Discovery page %s yielded zero new job IDs; stopping", page)
            stop_reason = "no_new_ids"
            break
        if max_jobs is not None and len(discovered_ids) >= max_jobs:
            logger.info("Discovery reached max_jobs=%s; stopping", max_jobs)
            stop_reason = "max_jobs"
            break

        page += 1

    db.set_state(conn, "discovery:last_page", str(last_processed_page))
    db.set_state(conn, "discovery:last_run_at", now_utc_iso())
    db.set_state(conn, "discovery:last_run_new_ids", str(len(discovered_ids)))
    db.set_state(conn, "discovery:last_completed", "1" if completed else "0")
    db.set_state(conn, "discovery:last_stop_reason", stop_reason or "")
    return DiscoveryResult(
        discovered_ids=discovered_ids,
        completed=completed,
        stop_reason=stop_reason,
        last_processed_page=last_processed_page,
    )
