from __future__ import annotations

from pathlib import Path

from euraxess_scraper.discovery import extract_job_links_from_html, extract_pagination_pages_from_html


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
