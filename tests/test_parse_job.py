from __future__ import annotations

from pathlib import Path

from euraxess_scraper.parse_job import parse_job_detail


def test_parse_job_fixture_extracts_expected_fields():
    html = Path("tests/fixtures/job_detail_sample.html").read_text(encoding="utf-8")
    record = parse_job_detail(html, "https://euraxess.ec.europa.eu/jobs/415163")

    assert record["job_id"] == "415163"
    assert record["position_type"] == "job_offer"
    assert record["title"]
    assert record["organization"]
    assert isinstance(record["sections_json"], str)
    assert record["cleaned_text"]
    assert record["content_hash"]
