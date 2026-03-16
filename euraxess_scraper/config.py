from __future__ import annotations

import logging
import os
from pathlib import Path

BASE_URL = "https://euraxess.ec.europa.eu"
SEARCH_URL = f"{BASE_URL}/jobs/search?f%5B0%5D=offer_type%3Ajob_offer"

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "euraxess.db"
EXPORT_DIR = DATA_DIR / "exports"
INDEX_DIR = DATA_DIR / "index"

USER_AGENT = (
    "euraxess-scraper/1.0 "
    "(+https://github.com/jpripamonti/euraxess-improved-search)"
)

DEFAULT_RPS = 1.0
DEFAULT_CONCURRENCY = 3

# httpx timeout fields: connect, read, write, pool
DEFAULT_TIMEOUTS = {
    "connect": 10.0,
    "read": 30.0,
    "write": 30.0,
    "pool": 30.0,
}

MAX_RETRIES = 5
BACKOFF = [1, 2, 4, 8, 16]
JITTER_MS_MAX = 300
GLOBAL_FAILURE_HALT = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
    )


def user_agent_has_placeholders() -> bool:
    return "YOUR_USERNAME" in USER_AGENT or "YOUR_EMAIL" in USER_AGENT
