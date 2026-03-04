from __future__ import annotations

from fastapi.testclient import TestClient

from euraxess_scraper import db
from euraxess_scraper.web.app import create_app


def _insert_job(conn, *, job_id: str, title: str, job_type_inferred: str):
    record = {column: None for column in db.JOB_COLUMNS}
    record.update(
        {
            "job_id": job_id,
            "url": f"https://euraxess.ec.europa.eu/jobs/{job_id}",
            "title": title,
            "organization": "Test Org",
            "country": "Germany",
            "cleaned_text": "example text",
            "sections_json": "{}",
            "content_hash": f"hash-{job_id}",
            "first_seen_at": "2026-03-01T00:00:00Z",
            "last_seen_at": "2026-03-01T00:00:00Z",
            "fetched_at": "2026-03-01T00:00:00Z",
            "http_status": 200,
            "deadline": "2099-01-01T00:00:00Z",
            "job_type_inferred": job_type_inferred,
            "job_type_score": 60,
        }
    )
    db.upsert_job_detail(conn, record)


def _seed_db(db_path):
    conn = db.get_connection(db_path)
    db.init_db(conn)
    _insert_job(conn, job_id="100", title="Postdoc in AI", job_type_inferred="postdoc")
    _insert_job(conn, job_id="101", title="PhD Candidate in ML", job_type_inferred="phd")
    conn.close()


def test_home_page_renders(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "EURAXESS Role Finder" in response.text


def test_search_page_applies_strict_type_filter(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    response = client.get(
        "/search",
        params={"q": "", "job_type": "phd", "active_only": "true", "open_only": "true"},
    )

    assert response.status_code == 200
    assert "PhD Candidate in ML" in response.text
    assert "Postdoc in AI" not in response.text


def test_api_search_returns_expected_json_shape(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    response = client.get("/api/search", params={"q": "", "job_type": "all", "page": 1, "page_size": 10})
    payload = response.json()

    assert response.status_code == 200
    assert {"query", "job_type", "page", "page_size", "total", "results", "has_next"} <= set(payload.keys())
    assert isinstance(payload["results"], list)
    assert payload["job_type"] == "all"

