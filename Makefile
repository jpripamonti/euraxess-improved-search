.PHONY: test integration crawl update stats

test:
	pytest

integration:
	pytest --integration -m integration

crawl:
	python -m euraxess_scraper.cli crawl

update:
	python -m euraxess_scraper.cli update

stats:
	python -m euraxess_scraper.cli stats
