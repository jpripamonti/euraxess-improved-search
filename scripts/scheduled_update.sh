#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Tunable defaults for periodic incremental refreshes.
MAX_PAGES="${MAX_PAGES:-30}"
RPS="${RPS:-0.4}"
CONCURRENCY="${CONCURRENCY:-2}"

python -m euraxess_scraper.cli update \
  --max-pages "${MAX_PAGES}" \
  --no-delist \
  --rps "${RPS}" \
  --concurrency "${CONCURRENCY}"

python -m euraxess_scraper.cli classify-topics --only-missing
python -m euraxess_scraper.cli build-index
python -m euraxess_scraper.cli export --format parquet --output data/exports/jobs.parquet
python -m euraxess_scraper.cli stats
