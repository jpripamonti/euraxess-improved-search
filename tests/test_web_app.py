from __future__ import annotations

import json

from fastapi.testclient import TestClient

from euraxess_scraper import db
from euraxess_scraper.web.app import create_app


def _insert_job(
    conn,
    *,
    job_id: str,
    title: str,
    job_type_inferred: str,
    topic_domain: str = "other",
    sections_json: str | None = None,
):
    record = {column: None for column in db.JOB_COLUMNS}
    record.update(
        {
            "job_id": job_id,
            "url": f"https://euraxess.ec.europa.eu/jobs/{job_id}",
            "title": title,
            "organization": "Test Org",
            "country": "Germany",
            "cleaned_text": "example text",
            "sections_json": sections_json if sections_json is not None else "{}",
            "content_hash": f"hash-{job_id}",
            "first_seen_at": "2026-03-01T00:00:00Z",
            "last_seen_at": "2026-03-01T00:00:00Z",
            "fetched_at": "2026-03-01T00:00:00Z",
            "http_status": 200,
            "deadline": "2099-01-01T00:00:00Z",
            "job_type_inferred": job_type_inferred,
            "job_type_score": 60,
            "topic_domain": topic_domain,
            "topic_confidence": 0.72,
            "topic_scores_json": json.dumps(
                {
                    "engineering_technology": 0.88,
                    "social_sciences": 0.11,
                    "other": 0.12,
                }
            ),
        }
    )
    db.upsert_job_detail(conn, record)


def _seed_db(db_path):
    conn = db.get_connection(db_path)
    db.init_db(conn)
    _insert_job(
        conn,
        job_id="100",
        title="Postdoc in AI",
        job_type_inferred="postdoc",
        topic_domain="engineering_technology",
        sections_json=json.dumps(
            {
                "offer_description": "Research on machine learning systems.",
                "requirements": "PhD in CS or related field.",
            }
        ),
    )
    _insert_job(
        conn,
        job_id="101",
        title="PhD Candidate in ML",
        job_type_inferred="phd",
        topic_domain="engineering_technology",
    )
    _insert_job(
        conn,
        job_id="102",
        title="PhD in Sociology",
        job_type_inferred="phd",
        topic_domain="social_sciences",
    )
    _insert_job(
        conn,
        job_id="103",
        title="General Research Assistant",
        job_type_inferred="unknown",
        topic_domain="other",
    )
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


def test_search_results_link_to_local_detail_page(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    response = client.get("/search", params={"q": "", "job_type": "all", "active_only": "true", "open_only": "true"})

    assert response.status_code == 200
    assert "/jobs/100" in response.text
    assert "Show description" in response.text


def test_job_detail_page_renders_sections(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    response = client.get("/jobs/100", params={"back": "/search?q=postdoc"})

    assert response.status_code == 200
    assert "Research on machine learning systems." in response.text
    assert "PhD in CS or related field." in response.text
    assert "Back to results" in response.text


def test_api_job_detail_returns_expected_shape(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    response = client.get("/api/jobs/100")
    payload = response.json()

    assert response.status_code == 200
    assert payload["job_id"] == "100"
    assert payload["topic_domain"] == "engineering_technology"
    assert "sections" in payload
    assert payload["sections"]["requirements"] == "PhD in CS or related field."


def test_topic_filter_applies_in_html_and_api(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    html = client.get("/search", params={"q": "", "topic": "social_sciences", "active_only": "true", "open_only": "true"})
    api = client.get("/api/search", params={"q": "", "topic": "social_sciences", "page": 1, "page_size": 20})

    assert html.status_code == 200
    assert "PhD in Sociology" in html.text
    assert "Postdoc in AI" not in html.text

    payload = api.json()
    assert api.status_code == 200
    assert payload["topic"] == "social_sciences"
    assert payload["results"]
    assert all(item["topic_domain"] == "social_sciences" for item in payload["results"])


def test_multi_topic_include_and_exclude_filters(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    payload = client.get(
        "/api/search",
        params=[
            ("q", ""),
            ("include_topic", "social_sciences"),
            ("include_topic", "other"),
            ("exclude_topic", "other"),
            ("page", "1"),
            ("page_size", "20"),
        ],
    ).json()

    assert payload["include_topics"] == ["social_sciences"]
    assert payload["exclude_topics"] == ["other"]
    assert payload["results"]
    assert all(item["topic_domain"] == "social_sciences" for item in payload["results"])


def test_api_search_returns_expected_json_shape(tmp_path):
    db_path = tmp_path / "app.db"
    _seed_db(db_path)
    app = create_app(db_path=db_path, index_dir=tmp_path / "index")
    client = TestClient(app)

    response = client.get("/api/search", params={"q": "", "job_type": "all", "page": 1, "page_size": 10})
    payload = response.json()

    assert response.status_code == 200
    assert {"query", "job_type", "topic", "page", "page_size", "total", "results", "has_next"} <= set(payload.keys())
    assert isinstance(payload["results"], list)
    assert payload["job_type"] == "all"
