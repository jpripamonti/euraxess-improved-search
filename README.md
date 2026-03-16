# EURAXESS Improved Search

A local search engine for academic job postings from [EURAXESS](https://euraxess.ec.europa.eu/jobs/search).

This project crawls the EURAXESS jobs listing into a local SQLite database, classifies postings by role and topic, builds a hybrid search index, and serves the results through a local FastAPI app. It is designed to run on your machine, with the data and index stored under `data/`.

## What it does

- Builds a local copy of EURAXESS job postings
- Supports incremental updates with HTTP caching headers
- Classifies jobs by role, topic domain, and language
- Provides hybrid search with SQLite FTS5 plus FAISS vectors
- Serves a local web UI and JSON API
- Exports data to JSONL or Parquet

## Project layout

```text
euraxess_scraper/
  cli.py
  config.py
  db.py
  discovery.py
  export.py
  fetch.py
  indexing.py
  language.py
  nli_classifier.py
  parse_job.py
  search.py
  taxonomy.py
  topics.py
  web/
    app.py
    templates/
  resources/
scripts/
  scheduled_update.sh
tests/
data/
  euraxess.db
  exports/
  index/
```

## Installation

Requirements: Python 3.11 or newer.

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

Optional environment override:

```bash
export DATA_DIR=/absolute/path/to/euraxess-data
```

By default, the app stores its database, exports, and vector index under `./data`.

## Build the local database

A full first-time crawl takes a while. Expect a few hours for the full dataset at the default crawl rate.

```bash
python -m euraxess_scraper.cli crawl
python -m euraxess_scraper.cli reclassify
python -m euraxess_scraper.cli classify-topics
python -m euraxess_scraper.cli detect-language
python -m euraxess_scraper.cli build-index
```

If you already have a populated `data/euraxess.db` and `data/index/`, you can skip straight to serving or updating.

## Run locally

### Web UI

```bash
python -m euraxess_scraper.cli serve --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### CLI search

```bash
python -m euraxess_scraper.cli search --query "machine learning postdoc germany"
python -m euraxess_scraper.cli search --query "quantum optics" --job-type postdoc
python -m euraxess_scraper.cli search --query "sociology phd" --topic social_sciences
python -m euraxess_scraper.cli search --query "bioinformatics" --country Germany
```

### Manual update

Fast incremental refresh:

```bash
python -m euraxess_scraper.cli update --max-pages 30 --no-delist
python -m euraxess_scraper.cli reclassify
python -m euraxess_scraper.cli classify-topics --only-missing
python -m euraxess_scraper.cli detect-language --only-missing
python -m euraxess_scraper.cli build-index
```

Full refresh:

```bash
python -m euraxess_scraper.cli update
python -m euraxess_scraper.cli reclassify
python -m euraxess_scraper.cli classify-topics --only-missing
python -m euraxess_scraper.cli detect-language --only-missing
python -m euraxess_scraper.cli build-index
```

Or use the helper script:

```bash
./scripts/scheduled_update.sh
```

You can wire that script into your own `cron` job or another local scheduler if you want recurring refreshes.

## Useful commands

```bash
# Crawl and update
python -m euraxess_scraper.cli crawl
python -m euraxess_scraper.cli crawl --dry-run
python -m euraxess_scraper.cli update
python -m euraxess_scraper.cli update --max-pages 30 --no-delist

# Classification
python -m euraxess_scraper.cli reclassify
python -m euraxess_scraper.cli nli-classify-type
python -m euraxess_scraper.cli classify-topics --only-missing
python -m euraxess_scraper.cli detect-language --only-missing

# Index and search
python -m euraxess_scraper.cli build-index
python -m euraxess_scraper.cli search --query "..."
python -m euraxess_scraper.cli eval-search --gold tests/fixtures/gold_queries.yaml

# Serve the local app
python -m euraxess_scraper.cli serve

# Data and exports
python -m euraxess_scraper.cli stats
python -m euraxess_scraper.cli export --format jsonl --output data/exports/jobs.jsonl
python -m euraxess_scraper.cli export --format parquet --output data/exports/jobs.parquet
```

## Web endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Search UI |
| `GET /search` | HTML search results |
| `GET /api/search` | JSON search API |
| `GET /jobs/{job_id}` | Job detail page |
| `GET /api/jobs/{job_id}` | Job detail as JSON |

Search parameters for `/search` and `/api/search`:

| Parameter | Type | Description |
|---|---|---|
| `q` | string | Free-text query |
| `job_type` | string | `all` \| `postdoc` \| `phd` \| `professor` \| `other` \| `unknown` |
| `include_topic` | string (repeatable) | Include jobs in these topic domains |
| `exclude_topic` | string (repeatable) | Exclude jobs in these topic domains |
| `country` | string | Country filter |
| `language` | string | ISO 639-1 code such as `en`, `fr`, or `de` |
| `page` | int | Page number |
| `page_size` | int | Results per page, max `100` |
| `active_only` | bool | Exclude delisted jobs |
| `open_only` | bool | Exclude past-deadline jobs |

Topic keys:
`computer_science`, `natural_sciences`, `engineering_technology`, `medical_health`, `agricultural_veterinary`, `social_sciences`, `humanities_arts`, `other`

## Rate limiting

The crawler is intentionally conservative:

- 1.0 request per second globally
- Up to 3 concurrent workers
- Random jitter between requests
- Retry backoff for transient failures
- Automatic halt after too many consecutive failures
- `robots.txt` check at startup

Lower the request rate further if you are doing long unattended runs.

## Testing

```bash
pytest
pytest --integration -m integration
```
