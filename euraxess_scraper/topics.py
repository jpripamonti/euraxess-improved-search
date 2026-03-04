from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .taxonomy import normalize_for_match
from .utils import clean_text

LOGGER = logging.getLogger(__name__)

DEFAULT_TOPIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOPIC_OTHER = "other"
TOPIC_MIN_SCORE = 0.20
TOPIC_MIN_MARGIN = 0.04
MAX_TOPIC_TEXT_CHARS = 6000


@dataclass(frozen=True)
class TopicClassification:
    topic_domain: str
    confidence: float
    scores: dict[str, float]


def _default_topics_text() -> str:
    return files("euraxess_scraper.resources").joinpath("topic_buckets.yaml").read_text(encoding="utf-8")


@lru_cache(maxsize=4)
def load_topic_domains(topics_path: str | None = None) -> dict[str, dict[str, Any]]:
    raw = Path(topics_path).read_text(encoding="utf-8") if topics_path else _default_topics_text()
    payload = yaml.safe_load(raw) or {}
    src = payload.get("topic_domains", {})
    domains: dict[str, dict[str, Any]] = {}
    for key, value in src.items():
        if not isinstance(key, str):
            continue
        item = value if isinstance(value, dict) else {}
        domains[key] = {
            "label": str(item.get("label") or key.replace("_", " ").title()),
            "description": str(item.get("description") or ""),
            "seed_terms": [str(x) for x in (item.get("seed_terms") or []) if str(x).strip()],
        }
    if TOPIC_OTHER not in domains:
        domains[TOPIC_OTHER] = {
            "label": "Other",
            "description": "Fallback when classification is uncertain.",
            "seed_terms": [],
        }
    return domains


def canonical_topic_domains(*, topics_path: str | None = None) -> tuple[str, ...]:
    return tuple(load_topic_domains(topics_path).keys())


def default_topic_labels(*, topics_path: str | None = None) -> dict[str, str]:
    labels = {"all": "All topics"}
    domains = load_topic_domains(topics_path)
    for domain, meta in domains.items():
        labels[domain] = str(meta.get("label") or domain)
    return labels


def normalize_topic_filter(value: str | None, *, topics_path: str | None = None) -> str | None:
    if value is None:
        return None
    normalized = normalize_for_match(value)
    if not normalized or normalized in {"all", "any"}:
        return None

    domains = load_topic_domains(topics_path)
    aliases: dict[str, str] = {}
    for domain, meta in domains.items():
        aliases[normalize_for_match(domain)] = domain
        aliases[normalize_for_match(str(meta.get("label") or domain))] = domain

    return aliases.get(normalized)


def normalize_topic_filters(
    values: list[str] | tuple[str, ...] | None,
    *,
    topics_path: str | None = None,
) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in str(raw or "").split(","):
            normalized = normalize_topic_filter(part, topics_path=topics_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
    return out


def build_topic_input_text(
    *,
    title: str | None,
    sections: dict[str, str] | None,
    cleaned_text: str | None,
    max_chars: int = MAX_TOPIC_TEXT_CHARS,
) -> str:
    sections = sections or {}
    chunks: list[str] = []
    if title:
        chunks.append(clean_text(title))
    for key in ("offer_description", "requirements", "additional_information", "work_locations"):
        value = sections.get(key)
        if value:
            chunks.append(clean_text(value))
    if cleaned_text:
        chunks.append(clean_text(cleaned_text))
    merged = "\n\n".join(chunk for chunk in chunks if chunk).strip()
    return merged[:max_chars]


def _topic_prototype_text(domain: str, meta: dict[str, Any]) -> str:
    label = str(meta.get("label") or domain)
    description = str(meta.get("description") or "")
    seeds = ", ".join([str(x) for x in (meta.get("seed_terms") or []) if str(x).strip()])
    return clean_text(f"{label}. {description}. Keywords: {seeds}")


@lru_cache(maxsize=4)
def _load_topic_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device="cpu")
    except TypeError:
        return SentenceTransformer(model_name)


def _encode_texts(texts: list[str], model_name: str):
    import numpy as np

    model = _load_topic_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


@lru_cache(maxsize=8)
def _prototype_matrix(model_name: str, topics_path: str | None):
    domains = load_topic_domains(topics_path)
    names: list[str] = []
    texts: list[str] = []
    for domain, meta in domains.items():
        if domain == TOPIC_OTHER:
            continue
        names.append(domain)
        texts.append(_topic_prototype_text(domain, meta))
    return names, _encode_texts(texts, model_name)


def classify_topic(
    *,
    title: str | None,
    sections: dict[str, str] | None,
    cleaned_text: str | None,
    model_name: str = DEFAULT_TOPIC_MODEL,
    topics_path: str | None = None,
    min_score: float = TOPIC_MIN_SCORE,
    min_margin: float = TOPIC_MIN_MARGIN,
) -> TopicClassification:
    import numpy as np

    domains = load_topic_domains(topics_path)
    score_map: dict[str, float] = {domain: 0.0 for domain in domains.keys()}
    text = build_topic_input_text(
        title=title,
        sections=sections,
        cleaned_text=cleaned_text,
    )
    if not text:
        return TopicClassification(topic_domain=TOPIC_OTHER, confidence=0.0, scores=score_map)

    try:
        query_vec = _encode_texts([text], model_name)[0]
        domain_names, prototypes = _prototype_matrix(model_name, topics_path)
        if not domain_names or prototypes.shape[0] == 0:
            return TopicClassification(topic_domain=TOPIC_OTHER, confidence=0.0, scores=score_map)
        sims = np.asarray(prototypes @ query_vec, dtype="float32")
    except Exception as exc:
        LOGGER.warning("Topic classification failed: %s", exc)
        return TopicClassification(topic_domain=TOPIC_OTHER, confidence=0.0, scores=score_map)

    ranked = sorted(zip(domain_names, sims.tolist()), key=lambda item: item[1], reverse=True)
    for domain, value in ranked:
        score_map[domain] = float(value)

    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = float(best_score - second_score)
    confidence = float(max(0.0, min(1.0, (best_score + 1.0) / 2.0)))
    score_map[TOPIC_OTHER] = float(max(0.0, min(1.0, 1.0 - confidence)))

    if float(best_score) < float(min_score) or margin < float(min_margin):
        return TopicClassification(
            topic_domain=TOPIC_OTHER,
            confidence=score_map[TOPIC_OTHER],
            scores=score_map,
        )

    return TopicClassification(topic_domain=best_domain, confidence=confidence, scores=score_map)


def classify_topics_batch(
    items: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_TOPIC_MODEL,
    topics_path: str | None = None,
    min_score: float = TOPIC_MIN_SCORE,
    min_margin: float = TOPIC_MIN_MARGIN,
    batch_size: int = 64,
) -> list[TopicClassification]:
    import numpy as np

    if not items:
        return []

    domains = load_topic_domains(topics_path)
    default_scores = {domain: 0.0 for domain in domains.keys()}
    texts = [
        build_topic_input_text(
            title=item.get("title"),
            sections=item.get("sections"),
            cleaned_text=item.get("cleaned_text"),
        )
        for item in items
    ]
    try:
        domain_names, prototypes = _prototype_matrix(model_name, topics_path)
        if not domain_names or prototypes.shape[0] == 0:
            return [TopicClassification(TOPIC_OTHER, 0.0, dict(default_scores)) for _ in items]
    except Exception as exc:
        LOGGER.warning("Topic batch prototype loading failed: %s", exc)
        return [TopicClassification(TOPIC_OTHER, 0.0, dict(default_scores)) for _ in items]

    out: list[TopicClassification] = []
    for start in range(0, len(texts), max(1, int(batch_size))):
        chunk = texts[start : start + max(1, int(batch_size))]
        vectors = None
        if any(chunk):
            try:
                vectors = _encode_texts(chunk, model_name)
            except Exception as exc:
                LOGGER.warning("Topic batch encoding failed: %s", exc)
                vectors = None

        for idx, text in enumerate(chunk):
            if not text or vectors is None:
                out.append(TopicClassification(TOPIC_OTHER, 0.0, dict(default_scores)))
                continue

            score_map = dict(default_scores)
            vec = vectors[idx]
            sims = np.asarray(prototypes @ vec, dtype="float32")
            ranked = sorted(zip(domain_names, sims.tolist()), key=lambda item: item[1], reverse=True)
            for domain, value in ranked:
                score_map[domain] = float(value)

            best_domain, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -1.0
            margin = float(best_score - second_score)
            confidence = float(max(0.0, min(1.0, (best_score + 1.0) / 2.0)))
            score_map[TOPIC_OTHER] = float(max(0.0, min(1.0, 1.0 - confidence)))
            if float(best_score) < float(min_score) or margin < float(min_margin):
                out.append(TopicClassification(TOPIC_OTHER, score_map[TOPIC_OTHER], score_map))
            else:
                out.append(TopicClassification(best_domain, confidence, score_map))
    return out
