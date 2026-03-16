from __future__ import annotations

import json

import numpy as np

from euraxess_scraper import topics


def _fake_encode_texts(texts, _model_name):
    vectors = []
    for text in texts:
        t = str(text).lower()
        if any(token in t for token in ["machine learning", "computer science", "informatique", "apprentissage automatique"]):
            vectors.append([1.0, 0.0, 0.0])
        elif any(token in t for token in ["engineering", "robotics", "automation"]):
            vectors.append([0.8, 0.2, 0.0])
        elif any(token in t for token in ["sociology", "economics", "social sciences", "science sociale"]):
            vectors.append([0.0, 1.0, 0.0])
        elif any(token in t for token in ["history", "philosophy", "humanities", "arts"]):
            vectors.append([0.0, 0.0, 1.0])
        else:
            vectors.append([0.1, 0.1, 0.1])
    return np.asarray(vectors, dtype="float32")


def test_multilingual_examples_map_to_expected_buckets(monkeypatch):
    topics._prototype_matrix.cache_clear()
    monkeypatch.setattr(topics, "_encode_texts", _fake_encode_texts)

    social = topics.classify_topic(
        title="Postdoctoral researcher in sociology",
        sections={},
        cleaned_text="Research on inequality and social institutions.",
        model_name="fake",
    )
    stem = topics.classify_topic(
        title="Bourse doctorale en apprentissage automatique et informatique",
        sections={},
        cleaned_text="Projet de recherche en intelligence artificielle.",
        model_name="fake",
    )

    assert social.topic_domain == "social_sciences"
    assert stem.topic_domain == "computer_science"


def test_ambiguous_case_falls_back_to_other(monkeypatch):
    topics._prototype_matrix.cache_clear()
    monkeypatch.setattr(topics, "_encode_texts", _fake_encode_texts)

    classified = topics.classify_topic(
        title="Interdisciplinary support role",
        sections={},
        cleaned_text="General tasks across departments.",
        model_name="fake",
    )

    assert classified.topic_domain == topics.TOPIC_OTHER


def test_batch_scores_are_json_serializable_and_complete(monkeypatch):
    topics._prototype_matrix.cache_clear()
    monkeypatch.setattr(topics, "_encode_texts", _fake_encode_texts)

    results = topics.classify_topics_batch(
        [
            {
                "title": "PhD in sociology",
                "sections": {"requirements": "social sciences background"},
                "cleaned_text": "qualitative methods",
            }
        ],
        model_name="fake",
        batch_size=2,
    )

    assert len(results) == 1
    score_map = results[0].scores
    assert set(topics.canonical_topic_domains()) <= set(score_map.keys())
    assert json.loads(json.dumps(score_map))
