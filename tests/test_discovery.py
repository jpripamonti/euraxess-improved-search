from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from euraxess_scraper import db
from euraxess_scraper.discovery import (
    discover_jobs,
    extract_job_links_from_html,
    extract_pagination_pages_from_html,
)
from euraxess_scraper.fetch import FetchResult


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    return conn


def test_extract_job_links_from_search_fixture():
    html = Path("tests/fixtures/search_results_page.html").read_text(encoding="utf-8")
    links = extract_job_links_from_html(html)

    assert links
    job_ids = [job_id for job_id, _ in links]
    assert len(job_ids) == len(set(job_ids))
    assert all(job_id.isdigit() for job_id in job_ids)


def test_extract_pagination_pages_from_search_fixture():
    html = Path("tests/fixtures/search_results_page.html").read_text(encoding="utf-8")
    pages = extract_pagination_pages_from_html(html)

    assert pages
    assert min(pages) >= 0
    assert pages == sorted(pages)


class _StubFetcher:
    def __init__(self):
        self.page_0_calls = 0

    async def get(self, url: str, **kwargs):
        if "/api/" in url or "_format=json" in url or "ajax=1" in url:
            return FetchResult(
                url=url,
                final_url=url,
                status=404,
                text="",
                etag=None,
                last_modified=None,
            )
        if "page=0" in url:
            self.page_0_calls += 1
            return FetchResult(
                url=url,
                final_url=url,
                status=429,
                text=None,
                etag=None,
                last_modified=None,
            )
        return FetchResult(
            url=url,
            final_url=url,
            status=404,
            text="",
            etag=None,
            last_modified=None,
        )


@pytest.mark.asyncio
async def test_discover_jobs_marks_partial_after_repeated_page_failures():
    conn = _conn()
    fetcher = _StubFetcher()

    result = await discover_jobs(
        conn,
        fetcher,
        max_page_failures=3,
        retry_cooldown_seconds=0,
    )

    assert result.completed is False
    assert result.stop_reason == "page_0_status_429"
    assert result.discovered_ids == set()
    assert fetcher.page_0_calls == 3
