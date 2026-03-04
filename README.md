# EURAXESS Scraper

Production-grade crawler and local hybrid-search pipeline for EURAXESS job offers.

- Discovery source: `https://euraxess.ec.europa.eu/jobs/search?f%5B0%5D=offer_type%3Ajob_offer`
- Storage: SQLite (`data/euraxess.db`)
- Search: SQLite FTS5 + FAISS + Sentence-Transformers (`all-MiniLM-L6-v2`)
- CLI: Typer (`python -m euraxess_scraper.cli ...`)

## Features

- Resumable queue-based crawl (`pending/done/failed`)
- Incremental updates with conditional requests (`If-None-Match` / `If-Modified-Since`)
- Resilient discovery retries with explicit partial-scan detection
- Deduplication by canonical EURAXESS job ID (`/jobs/<numeric_id>`)
- Delisting tracking (`delisted_at`, no hard deletes)
- Inferred strict role taxonomy (`postdoc`, `phd`, `professor`, `unknown`)
- In-app job detail pages backed by parsed local content
- AI topic classification with broad domains (`natural_sciences`, `engineering_technology`, `medical_health`, `agricultural_veterinary`, `social_sciences`, `humanities_arts`, `other`)
- Export to JSONL and Parquet
- Weighted hybrid keyword + semantic search using Reciprocal Rank Fusion (RRF)
- Deterministic reranking layer for improved relevance
- Search evaluation command with `nDCG/MRR/Recall` on gold queries
- FastAPI web UI + JSON API for local/self-hosted search

## Project Layout

- `euraxess_scraper/cli.py`: Typer commands (`crawl`, `update`, `export`, `build-index`, `reclassify`, `search`, `stats`, `serve`)
- `euraxess_scraper/discovery.py`: listing pagination + lightweight endpoint probe
- `euraxess_scraper/fetch.py`: async HTTP fetcher, retry/backoff, global halt guard
- `euraxess_scraper/parse_job.py`: detail page parser
- `euraxess_scraper/taxonomy.py`: role inference + synonym expansion
- `euraxess_scraper/topics.py`: topic-domain inference + labels
- `euraxess_scraper/db.py`: schema + DB helpers
- `euraxess_scraper/indexing.py`: FTS5 + FAISS build
- `euraxess_scraper/search.py`: hybrid retrieval and RRF merge
- `euraxess_scraper/export.py`: JSONL / Parquet export
- `euraxess_scraper/web/`: FastAPI app + Jinja templates
- `euraxess_scraper/resources/topic_buckets.yaml`: topic-domain taxonomy config
- `scripts/scheduled_update.sh`: incremental refresh for cron/launchd
- `tests/`: unit + integration tests

## Installation

### Option A: `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"
```

## Usage

### 1) Crawl

```bash
python -m euraxess_scraper.cli crawl
```

Useful options:

```bash
python -m euraxess_scraper.cli crawl --dry-run
python -m euraxess_scraper.cli crawl --limit 10
python -m euraxess_scraper.cli crawl --max-pages 30
python -m euraxess_scraper.cli crawl --rps 1.0 --concurrency 3
python -m euraxess_scraper.cli crawl --verbose
```

### 2) Update

```bash
python -m euraxess_scraper.cli update --rps 1.0 --concurrency 3
python -m euraxess_scraper.cli update --max-pages 30 --no-delist --rps 0.4 --concurrency 2
```

Notes:
- Use `--max-pages` + `--no-delist` for fast incremental runs (new jobs from newest pages).
- Use full `update` (without `--max-pages`) when you want reliable delisting.

### 3) Export

```bash
python -m euraxess_scraper.cli export --format jsonl --output data/exports/jobs.jsonl
python -m euraxess_scraper.cli export --format parquet --output data/exports/jobs.parquet
```

### 4) Build indexes

```bash
python -m euraxess_scraper.cli build-index
python -m euraxess_scraper.cli build-index --model all-MiniLM-L6-v2 --batch-size 64
```

### 5) Hybrid search

```bash
python -m euraxess_scraper.cli search --query "machine learning postdoc germany" --top-k 10
python -m euraxess_scraper.cli search --query "data science" --country Germany
python -m euraxess_scraper.cli search --query "quantum optics" --job-type postdoc
python -m euraxess_scraper.cli search --query "sociology phd" --topic social_sciences
python -m euraxess_scraper.cli search --query "computational biology" --vector-weight 3.0 --keyword-weight 1.0
python -m euraxess_scraper.cli search --query "bioinformatics phd" --semantic-only
python -m euraxess_scraper.cli search --query "assistant professor economics" --no-rerank
```

### 6) Reclassify role taxonomy

```bash
python -m euraxess_scraper.cli reclassify
```

### 7) Classify topic domains

```bash
python -m euraxess_scraper.cli classify-topics --only-missing
python -m euraxess_scraper.cli classify-topics --since 2026-03-01T00:00:00Z
```

### 8) Evaluate search quality

```bash
python -m euraxess_scraper.cli eval-search --gold tests/fixtures/gold_queries.yaml
python -m euraxess_scraper.cli eval-search --gold tests/fixtures/gold_queries.yaml --min-ndcg-gain 0.05
```

### 9) Web app (FastAPI)

```bash
python -m euraxess_scraper.cli serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

Available web endpoints:
- `GET /`: search UI
- `GET /search`: HTML results endpoint (`q`, `job_type`, `topic`, `country`, `page`, `page_size`, `active_only`, `open_only`, `debug`)
- `GET /jobs/{job_id}`: local detail page for a single job
- `GET /api/search`: JSON API (`q`, `job_type`, `topic`, `country`, `page`, `page_size`, `active_only`, `open_only`, `debug`)
- `GET /api/jobs/{job_id}`: JSON detail payload for one job

### 10) Stats

```bash
python -m euraxess_scraper.cli stats
```

### 11) Recurring incremental updates (every few days)

Run manually:

```bash
./scripts/scheduled_update.sh
```

Example cron entry (every 3 days at 06:00 local time):

```bash
0 6 */3 * * cd /Users/neo/Repos/Projects/euraxess_scraping && /bin/bash scripts/scheduled_update.sh >> data/exports/scheduled_update.log 2>&1
```

## Data Artifacts

- DB: `data/euraxess.db`
- Exports: `data/exports/`
- Vector index: `data/index/faiss.index`
- Vector mapping: `data/index/faiss_mapping.json`
- Vector matrix: `data/index/vectors.npy`

## Rate Limiting / Politeness

Defaults:

- `1.0` request/sec globally
- up to `3` concurrent workers
- random jitter `0-300ms`
- retries with backoff: `1, 2, 4, 8, 16` seconds
- max per-URL attempts: `5`
- global stop: halt after `50` consecutive request failures

`robots.txt` is checked at startup. If disallowed, the CLI logs a warning and continues.

## Crawl Duration Estimate

At `1 req/sec` and `3` workers (roughly ~`3 req/sec` effective), assuming ~30,000 jobs:

- Discovery: ~300 pages x 0.33s ≈ 2 minutes
- Detail crawl: ~30,000 jobs x 0.33s ≈ 2.75 hours
- Total first run: ≈ 3 hours
- `update` runs: much faster (conditional requests + new content only)

## Testing

Unit tests:

```bash
pytest
```

Integration test (live EURAXESS, explicitly enabled):

```bash
pytest --integration -m integration
```

## Notes

The default `USER_AGENT` contains placeholders in `euraxess_scraper/config.py`:

```text
euraxess-scraper/1.0 (+https://github.com/YOUR_USERNAME/euraxess-scraper; contact: YOUR_EMAIL)
```

Update `YOUR_USERNAME` and `YOUR_EMAIL` before heavy crawls.
