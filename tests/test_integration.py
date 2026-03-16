from __future__ import annotations

import sqlite3

import pytest

from euraxess_scraper import db, discovery
from euraxess_scraper.fetch import Fetcher


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_discover_first_two_pages_only():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)

    async with Fetcher(rps=1.0) as fetcher:
        discovered = await discovery.discover_jobs(
            conn,
            fetcher,
            requeue_existing=False,
            max_pages=2,
        )

    assert discovered
    assert len(discovered) == len(set(discovered))
