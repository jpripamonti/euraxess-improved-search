from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from .config import BASE_URL

JOB_URL_RE = re.compile(r"/jobs/(\d+)(?:$|[/?#])")


def now_utc_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def canonicalize_url(url: str, base_url: str = BASE_URL) -> str:
    absolute = urljoin(base_url, url)
    parsed = urlparse(absolute)
    # Keep query for listing pages, drop query for detail pages.
    if JOB_URL_RE.search(parsed.path):
        parsed = parsed._replace(query="")
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


def extract_job_id(url: str) -> str | None:
    match = JOB_URL_RE.search(urlparse(url).path)
    return match.group(1) if match else None


def dedupe_id_from_url(url: str) -> str:
    canonical = canonicalize_url(url)
    job_id = extract_job_id(canonical)
    if job_id:
        return job_id
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_http_date_to_utc_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_date_to_utc_iso(value: str | None) -> str | None:
    if not value:
        return None
    text = clean_text(value)
    if not text:
        return None

    # Prefer full ISO timestamps.
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass

    # Trim trailing timezone labels from strings like "16 Mar 2026 - 23:59 (Atlantic/Canary)".
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)

    patterns = [
        "%d %b %Y - %H:%M",
        "%d %B %Y - %H:%M",
        "%d %b %Y",
        "%d %B %Y",
        "%Y-%m-%d",
    ]
    for pattern in patterns:
        try:
            dt = datetime.strptime(text, pattern).replace(tzinfo=UTC)
            return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    return None


def robots_diagnostic(target_url: str, user_agent: str, base_url: str = BASE_URL) -> dict:
    robots_url = urljoin(base_url, "/robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
        allowed = parser.can_fetch(user_agent, target_url)
        return {"robots_url": robots_url, "allowed": allowed, "error": None}
    except Exception as exc:  # pragma: no cover - defensive
        return {"robots_url": robots_url, "allowed": True, "error": str(exc)}


def extract_page_numbers_from_links(hrefs: Iterable[str]) -> list[int]:
    pages: set[int] = set()
    for href in hrefs:
        query = parse_qs(urlparse(href).query)
        if "page" not in query or not query["page"]:
            continue
        try:
            pages.add(int(query["page"][0]))
        except ValueError:
            continue
    return sorted(pages)
