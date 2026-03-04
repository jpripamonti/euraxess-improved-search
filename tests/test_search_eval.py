from __future__ import annotations

from euraxess_scraper import cli


def test_parse_gold_cases_reads_queries(tmp_path):
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        """
queries:
  - query: "postdoc machine learning"
    relevant:
      "100": 3
      "101": 1
    job_type: postdoc
  - query: "invalid case"
    relevant: {}
        """.strip(),
        encoding="utf-8",
    )

    cases = cli._parse_gold_cases(gold)
    assert len(cases) == 1
    assert cases[0]["query"] == "postdoc machine learning"
    assert cases[0]["relevant"]["100"] == 3.0


def test_evaluate_mode_computes_metrics(monkeypatch):
    def fake_hybrid_search(_conn, **kwargs):
        if kwargs["query"] == "postdoc machine learning":
            return [{"job_id": "100"}, {"job_id": "999"}]
        return []

    monkeypatch.setattr(cli.search_mod, "hybrid_search", fake_hybrid_search)

    cases = [
        {
            "query": "postdoc machine learning",
            "relevant": {"100": 3.0, "101": 1.0},
            "job_type": None,
            "topic": None,
            "country": None,
            "active_only": True,
            "open_only": True,
        }
    ]
    metrics = cli._evaluate_mode(None, cases, mode="hybrid_no_rerank", k=2)

    assert metrics["queries"] == 1.0
    assert metrics["mrr@k"] == 1.0
    assert metrics["recall@k"] == 0.5
    assert 0.91 < metrics["ndcg@k"] < 0.93
