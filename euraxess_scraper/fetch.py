from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from time import monotonic
from typing import Mapping

import httpx

from . import config


class GlobalHaltError(RuntimeError):
    """Raised when global consecutive request failures exceed the configured threshold."""


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str | None
    etag: str | None
    last_modified: str | None
    error: str | None = None
    attempts: int = 1


class GlobalFailureTracker:
    def __init__(self, max_failures: int = config.GLOBAL_FAILURE_HALT):
        self.max_failures = max_failures
        self.consecutive_failures = 0
        self._lock = asyncio.Lock()

    async def success(self) -> None:
        async with self._lock:
            self.consecutive_failures = 0

    async def failure(self) -> None:
        async with self._lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                raise GlobalHaltError(
                    f"Halting crawl after {self.consecutive_failures} consecutive request failures."
                )


class AsyncRateLimiter:
    def __init__(self, rps: float, jitter_ms_max: int = config.JITTER_MS_MAX):
        if rps <= 0:
            raise ValueError("rps must be > 0")
        self.rps = rps
        self.jitter_ms_max = jitter_ms_max
        self._lock = asyncio.Lock()
        self._next_allowed = monotonic()

    async def acquire(self) -> None:
        async with self._lock:
            now = monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + (1.0 / self.rps)
        jitter = random.uniform(0, self.jitter_ms_max) / 1000.0
        delay = wait + jitter
        if delay > 0:
            await asyncio.sleep(delay)


class Fetcher:
    def __init__(
        self,
        *,
        rps: float,
        timeout: Mapping[str, float] | None = None,
        max_retries: int = config.MAX_RETRIES,
        backoff: list[int] | None = None,
        jitter_ms_max: int = config.JITTER_MS_MAX,
        user_agent: str = config.USER_AGENT,
    ):
        self.rps = rps
        self.max_retries = max_retries
        self.backoff = backoff or config.BACKOFF
        self.rate_limiter = AsyncRateLimiter(rps=rps, jitter_ms_max=jitter_ms_max)
        self.failure_tracker = GlobalFailureTracker(max_failures=config.GLOBAL_FAILURE_HALT)
        self.timeout = httpx.Timeout(**(timeout or config.DEFAULT_TIMEOUTS))
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": config.BASE_URL + "/",
            "Connection": "keep-alive",
        }
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Fetcher":
        self.client = httpx.AsyncClient(
            http2=True,
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.client is not None:
            await self.client.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> FetchResult:
        if self.client is None:
            raise RuntimeError("Fetcher must be used with async context manager")

        merged_headers: dict[str, str] = {}
        if headers:
            merged_headers.update(headers)

        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            await self.rate_limiter.acquire()
            try:
                response = await self.client.request(method, url, headers=merged_headers)
                status = response.status_code

                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    await self.failure_tracker.failure()
                    await asyncio.sleep(self.backoff[min(attempt - 1, len(self.backoff) - 1)])
                    continue

                if status in {429, 500, 502, 503, 504}:
                    await self.failure_tracker.failure()
                else:
                    await self.failure_tracker.success()

                return FetchResult(
                    url=url,
                    final_url=str(response.url),
                    status=status,
                    text=response.text if status != 304 else None,
                    etag=response.headers.get("etag"),
                    last_modified=response.headers.get("last-modified"),
                    error=None,
                    attempts=attempt,
                )

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                await self.failure_tracker.failure()
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff[min(attempt - 1, len(self.backoff) - 1)])
                    continue

        return FetchResult(
            url=url,
            final_url=url,
            status=0,
            text=None,
            etag=None,
            last_modified=None,
            error=last_error or "request failed",
            attempts=self.max_retries,
        )

    async def get(self, url: str, *, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        conditional_headers: dict[str, str] = {}
        if etag:
            conditional_headers["If-None-Match"] = etag
        if last_modified:
            conditional_headers["If-Modified-Since"] = last_modified
        return await self.request("GET", url, headers=conditional_headers)
