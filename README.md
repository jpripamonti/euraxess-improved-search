# EURAXESS Scraper

Production-grade crawler and local hybrid-search pipeline for EURAXESS job postings.

- **Source:** `https://euraxess.ec.europa.eu/jobs/search?f%5B0%5D=offer_type%3Ajob_offer`
- **Storage:** SQLite (`data/euraxess.db`)
- **Search:** SQLite FTS5 + FAISS + Sentence-Transformers (`all-MiniLM-L6-v2`)
- **CLI:** Typer (`python -m euraxess_scraper.cli ...`)

## Features

### Crawling & Storage
- Resumable queue-based crawl (`pending / done / failed`)
- Incremental updates with conditional HTTP requests (`If-None-Match` / `If-Modified-Since`)
- Resilient discovery retries with explicit partial-scan detection
- Deduplication by canonical EURAXESS job ID (`/jobs/<numeric_id>`)
- Delisting tracking (`delisted_at`, no hard deletes)
- Export to JSONL and Parquet

### Classification
- **Role taxonomy** — keyword + profile-signal classifier assigns each job to one of:
  - `postdoc` — postdoctoral fellowships
  - `phd` — doctoral / PhD positions
  - `professor` — faculty, lecturer, associate/full professor openings
  - `other` — research support & technical staff (Other Profession profile)
  - `unknown` — open calls for any career level (all-4-profiles)
- **NLI fallback** — zero-shot NLI classifier (`cross-encoder/nli-deberta-v3-small`) resolves remaining unknowns without an API key
- **Topic domains** — multilingual prototype-matching classifier (`paraphrase-multilingual-MiniLM-L12-v2`) assigns each job to a broad academic domain:
  - Computer Science & AI
  - Natural Sciences
  - Engineering & Technology
  - Medical & Health Sciences
  - Agricultural & Veterinary Sciences
  - Social Sciences
  - Humanities & Arts
  - Other / Interdisciplinary
- **Language detection** — `langdetect` identifies posting language (en, fr, de, es, …); stored per job and filterable in search

### Search & UI
- Weighted hybrid search: keyword (FTS5) + semantic (FAISS) merged with Reciprocal Rank Fusion (RRF)
- Deterministic reranking layer for improved relevance
- FastAPI web UI with faceted filters: role type, topic (multi-select include/exclude), country, language
- Confidence bars on every result card — role score and topic confidence normalized 0–100 with red→yellow→green gradient
- In-app job detail pages backed by locally parsed content
- JSON API for programmatic access
- Search evaluation command with nDCG / MRR / Recall on gold queries

## Project Layout

```
euraxess_scraper/
├── cli.py                          # Typer CLI (all commands)
├── discovery.py                    # Listing pagination + endpoint probe
├── fetch.py                        # Async HTTP fetcher, retry/backoff, halt guard
├── parse_job.py                    # Detail page HTML parser
├── taxonomy.py                     # Role inference, synonym expansion, type labels
├── topics.py                       # Topic-domain prototype classifier
├── language.py                     # Language detection (langdetect wrapper)
├── nli_classifier.py               # Zero-shot NLI job-type classifier
├── db.py                           # SQLite schema, queries, facet helpers
├── indexing.py                     # FTS5 + FAISS index builder
├── search.py                       # Hybrid retrieval and RRF merge
├── export.py                       # JSONL / Parquet export
├── web/
│   ├── app.py                      # FastAPI application
│   └── templates/
│       ├── index.html              # Search UI
│       └── job_detail.html         # Job detail page
└── resources/
    ├── topic_buckets.yaml          # Topic domain taxonomy (seed terms per domain)
    └── job_type_synonyms.yaml      # Keyword synonyms for role classification
scripts/
└── scheduled_update.sh             # Incremental refresh script for cron/launchd
tests/                              # Unit + integration tests
data/
├── euraxess.db                     # SQLite database
├── exports/                        # JSONL / Parquet exports
└── index/                          # FAISS vector index + mapping files
```

## Installation

### Option A: `uv` (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"   # dev dependencies (pytest)
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.11

## Usage

### 1. Crawl (first run)

```bash
python -m euraxess_scraper.cli crawl
```

Useful options:

```bash
python -m euraxess_scraper.cli crawl --dry-run            # discover only, no detail fetch
python -m euraxess_scraper.cli crawl --limit 50           # fetch at most 50 job details
python -m euraxess_scraper.cli crawl --max-pages 30       # limit discovery to 30 pages
python -m euraxess_scraper.cli crawl --rps 1.0 --concurrency 3
python -m euraxess_scraper.cli crawl --verbose
```

### 2. Update (incremental)

```bash
python -m euraxess_scraper.cli update --rps 1.0 --concurrency 3
```

Useful options:

```bash
# Fast incremental: only new jobs from the first N pages, skip delisting
python -m euraxess_scraper.cli update --max-pages 30 --no-delist

# Full scan with reliable delisting (slower but thorough)
python -m euraxess_scraper.cli update
```

> **Tip:** Use `--max-pages` + `--no-delist` for daily quick runs. Run a full `update` weekly to reliably detect delisted postings.

### 3. Classify roles

```bash
# Keyword + profile-signal classifier (fast, no model download)
python -m euraxess_scraper.cli reclassify

# Zero-shot NLI fallback for remaining unknowns (~180 MB model, cached)
python -m euraxess_scraper.cli nli-classify-type
python -m euraxess_scraper.cli nli-classify-type --all               # reclassify everything
python -m euraxess_scraper.cli nli-classify-type --min-confidence 0.6
```

### 4. Classify topic domains

```bash
python -m euraxess_scraper.cli classify-topics
python -m euraxess_scraper.cli classify-topics --only-missing        # skip already classified
python -m euraxess_scraper.cli classify-topics --since 2026-03-01T00:00:00Z
python -m euraxess_scraper.cli classify-topics --batch-size 128
```

### 5. Detect language

```bash
python -m euraxess_scraper.cli detect-language
python -m euraxess_scraper.cli detect-language --only-missing        # skip already detected
python -m euraxess_scraper.cli detect-language --limit 500
```

### 6. Build search indexes

```bash
python -m euraxess_scraper.cli build-index
python -m euraxess_scraper.cli build-index --model all-MiniLM-L6-v2 --batch-size 64
```

### 7. Hybrid search (CLI)

```bash
python -m euraxess_scraper.cli search --query "machine learning postdoc germany"
python -m euraxess_scraper.cli search --query "data science" --country Germany
python -m euraxess_scraper.cli search --query "quantum optics" --job-type postdoc
python -m euraxess_scraper.cli search --query "sociology phd" --topic social_sciences
python -m euraxess_scraper.cli search --query "bioinformatics" --vector-weight 3.0 --keyword-weight 1.0
python -m euraxess_scraper.cli search --query "computational biology" --semantic-only
python -m euraxess_scraper.cli search --query "assistant professor economics" --no-rerank
python -m euraxess_scraper.cli search --query "immunology" --top-k 20 --debug
```

### 8. Evaluate search quality

```bash
python -m euraxess_scraper.cli eval-search --gold tests/fixtures/gold_queries.yaml
python -m euraxess_scraper.cli eval-search --gold tests/fixtures/gold_queries.yaml --min-ndcg-gain 0.05
```

### 9. Web app

```bash
python -m euraxess_scraper.cli serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

#### Web endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Search UI |
| `GET /search` | HTML search results |
| `GET /api/search` | JSON search API |
| `GET /jobs/{job_id}` | Job detail page (HTML) |
| `GET /api/jobs/{job_id}` | Job detail (JSON) |

#### Search query parameters (both `/search` and `/api/search`)

| Parameter | Type | Description |
|---|---|---|
| `q` | string | Free-text search query |
| `job_type` | string | `all` \| `postdoc` \| `phd` \| `professor` \| `other` \| `unknown` |
| `include_topic` | string (repeatable) | Include jobs matching any of these topic keys |
| `exclude_topic` | string (repeatable) | Exclude jobs matching any of these topic keys |
| `country` | string | Country name filter |
| `language` | string | ISO 639-1 code: `en`, `fr`, `de`, `es`, … |
| `page` | int | Page number (default: 1) |
| `page_size` | int | Results per page (default: 20, max: 100) |
| `active_only` | bool | Exclude delisted jobs (default: true) |
| `open_only` | bool | Exclude past-deadline jobs (default: true) |
| `debug` | bool | Include score component diagnostics |

Topic keys: `computer_science`, `natural_sciences`, `engineering_technology`, `medical_health`, `agricultural_veterinary`, `social_sciences`, `humanities_arts`, `other`

### 10. Stats

```bash
python -m euraxess_scraper.cli stats
```

### 11. Export

```bash
python -m euraxess_scraper.cli export --format jsonl --output data/exports/jobs.jsonl
python -m euraxess_scraper.cli export --format parquet --output data/exports/jobs.parquet
```

### 12. Automated daily updates (GitHub Actions)

The included workflow (`.github/workflows/daily-update.yml`) keeps the database fresh automatically — no server required.

**Schedule:**
- **Mon–Sat 04:00 UTC** — fast incremental update (first 30 listing pages, new jobs only)
- **Sunday 03:00 UTC** — full scan with delisting check

The database and search index are stored as assets on a rolling GitHub Release (`db-latest`) and replaced on every run.

#### First-time setup

1. Push the repo to GitHub (see [Renaming / pushing](#renaming--pushing) below)
2. Upload your existing database to the `db-latest` release:

```bash
gh release create db-latest \
  data/euraxess.db \
  data/index/faiss.index \
  data/index/faiss_mapping.json \
  data/index/vectors.npy \
  --title "Database: initial" \
  --notes "Initial database upload" \
  --prerelease
```

3. GitHub Actions will take it from there — downloading, updating, and re-uploading on schedule.

> **No secrets needed.** The workflow uses the built-in `GITHUB_TOKEN` which has the required `contents: write` permission.

#### Manual trigger

Go to **Actions → Daily EURAXESS Update → Run workflow** to trigger a run at any time.

#### Download the latest database locally

```bash
gh release download db-latest \
  --repo <your-username>/<your-repo> \
  --dir data/tmp/
mv data/tmp/euraxess.db data/
mv data/tmp/faiss.index data/tmp/faiss_mapping.json data/tmp/vectors.npy data/index/
```

### 13. Recurring updates (local cron / launchd)

If you prefer to run updates on your own machine:

```bash
./scripts/scheduled_update.sh
```

Example cron entry (every 3 days at 06:00 local time):

```bash
0 6 */3 * * cd /path/to/euraxess_scraping && /bin/bash scripts/scheduled_update.sh >> data/exports/scheduled_update.log 2>&1
```

The script runs `update`, `reclassify`, `classify-topics`, `detect-language`, and `build-index` in sequence.

## Classification Details

### Role taxonomy

The keyword classifier assigns scores based on:

| Signal | Weight |
|---|---|
| Title keyword match | 70 |
| Body text keyword match | 12 |
| EURAXESS researcher profile boost | up to 45 |

Minimum score to assign a label: **25**. Maximum practical score: **127** (displayed as 0–100 in the UI).

Synonyms and keywords are configured in `euraxess_scraper/resources/job_type_synonyms.yaml`. Includes French academic terms (ATER → postdoc, doctorat → PhD).

### Topic domains

Uses `paraphrase-multilingual-MiniLM-L12-v2` to embed each job and compare cosine similarity against prototype embeddings built from seed terms in `topic_buckets.yaml`.

- Minimum score to assign a domain: **0.20**
- Minimum margin over the second-best domain: **0.02**
- Jobs below threshold are labelled "Other / Interdisciplinary"

### Language detection

Uses `langdetect` on the job's cleaned text. Stored as ISO 639-1 code (e.g. `en`, `fr`, `de`). Run `detect-language` after each crawl/update.

## Data Artifacts

| Path | Description |
|---|---|
| `data/euraxess.db` | SQLite database |
| `data/exports/` | JSONL / Parquet exports |
| `data/index/faiss.index` | FAISS vector index |
| `data/index/faiss_mapping.json` | Vector ID → job ID mapping |
| `data/index/vectors.npy` | Raw embedding matrix |

## Rate Limiting / Politeness

Defaults:

- `1.0` request/sec globally
- Up to `3` concurrent workers
- Random jitter: 0–300 ms
- Retry backoff: 1, 2, 4, 8, 16 seconds
- Max attempts per URL: 5
- Global stop: halt after 50 consecutive request failures

`robots.txt` is checked at startup; a warning is logged if the crawler path is disallowed.

## Crawl Duration Estimate

At 1 req/sec with 3 workers (~3 req/sec effective), for ~30,000 jobs:

| Phase | Estimate |
|---|---|
| Discovery (300 pages) | ~2 minutes |
| Detail crawl (30,000 jobs) | ~2.75 hours |
| **Total (first run)** | **~3 hours** |
| Incremental `update` | Much faster (conditional requests) |

## Testing

```bash
# Unit tests
pytest

# Integration tests (hits live EURAXESS — use sparingly)
pytest --integration -m integration
```

## Configuration

The default `USER_AGENT` in `euraxess_scraper/config.py` contains placeholders:

```
euraxess-scraper/1.0 (+https://github.com/YOUR_USERNAME/euraxess-scraper; contact: YOUR_EMAIL)
```

Update `YOUR_USERNAME` and `YOUR_EMAIL` before running heavy crawls.
