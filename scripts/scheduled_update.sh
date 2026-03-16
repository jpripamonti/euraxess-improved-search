#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Tunable defaults for periodic incremental refreshes.
MAX_PAGES="${MAX_PAGES:-30}"
RPS="${RPS:-0.4}"
CONCURRENCY="${CONCURRENCY:-2}"

echo "[1/6] Updating jobs (max ${MAX_PAGES} pages)..."
python -m euraxess_scraper.cli update \
  --max-pages "${MAX_PAGES}" \
  --no-delist \
  --rps "${RPS}" \
  --concurrency "${CONCURRENCY}"

echo "[2/6] Classifying roles..."
python -m euraxess_scraper.cli reclassify

echo "[3/6] Classifying topics..."
python -m euraxess_scraper.cli classify-topics --only-missing

echo "[4/6] Detecting languages..."
python -m euraxess_scraper.cli detect-language --only-missing

echo "[5/6] Building search index..."
python -m euraxess_scraper.cli build-index

echo "[6/6] Stats..."
python -m euraxess_scraper.cli stats
