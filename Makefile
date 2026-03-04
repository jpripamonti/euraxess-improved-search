.PHONY: test integration crawl update update-recent build-index export stats sync-recent sync-full

test:
	pytest

integration:
	pytest --integration -m integration

crawl:
	python -m euraxess_scraper.cli crawl

update:
	python -m euraxess_scraper.cli update

update-recent:
	python -m euraxess_scraper.cli update --max-pages 30 --no-delist --rps 0.4 --concurrency 2

build-index:
	python -m euraxess_scraper.cli build-index

export:
	python -m euraxess_scraper.cli export --format parquet --output data/exports/jobs.parquet

sync-recent: update-recent build-index export stats

sync-full:
	python -m euraxess_scraper.cli update --rps 0.4 --concurrency 2 --delist
	python -m euraxess_scraper.cli build-index
	python -m euraxess_scraper.cli export --format parquet --output data/exports/jobs.parquet
	python -m euraxess_scraper.cli stats

stats:
	python -m euraxess_scraper.cli stats
