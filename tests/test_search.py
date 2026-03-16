from __future__ import annotations

from euraxess_scraper.search import rrf_merge


def test_rrf_merge_can_prioritize_semantic_results():
    keyword_ranked = [("job_a", 1), ("job_b", 2)]
    semantic_ranked = [("job_b", 1), ("job_a", 2)]

    merged = rrf_merge(
        keyword_ranked,
        semantic_ranked,
        k=60,
        keyword_weight=1.0,
        vector_weight=3.0,
    )

    assert merged[0][0] == "job_b"


def test_rrf_merge_handles_semantic_only_input():
    merged = rrf_merge(
        [],
        [("job_semantic", 1)],
        k=60,
        keyword_weight=1.0,
        vector_weight=2.0,
    )

    assert merged == [("job_semantic", 2.0 / 61.0)]
