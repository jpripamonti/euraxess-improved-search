from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

CANONICAL_JOB_TYPES: tuple[str, ...] = ("postdoc", "phd", "professor", "unknown")

TITLE_WEIGHT = 70
TEXT_WEIGHT = 12
MIN_SCORE = 35
MIN_MARGIN = 12


@dataclass(frozen=True)
class JobTypeClassification:
    job_type: str
    score: int
    score_breakdown: dict[str, int]


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_for_match(value: str | None) -> str:
    if not value:
        return ""
    text = _strip_accents(value.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _default_synonyms_text() -> str:
    return files("euraxess_scraper.resources").joinpath("job_type_synonyms.yaml").read_text(
        encoding="utf-8"
    )


def _normalize_aliases(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_for_match(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


@lru_cache(maxsize=4)
def load_synonyms(synonyms_path: str | None = None) -> dict[str, list[str]]:
    if synonyms_path:
        raw = Path(synonyms_path).read_text(encoding="utf-8")
    else:
        raw = _default_synonyms_text()
    payload = yaml.safe_load(raw) or {}
    src = payload.get("job_types", {})

    aliases: dict[str, list[str]] = {}
    for key in ("postdoc", "phd", "professor"):
        values = src.get(key) or [key]
        if key not in values:
            values.append(key)
        aliases[key] = _normalize_aliases([str(v) for v in values])
    return aliases


def _matches_alias(text: str, alias: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _profile_boosts(researcher_profile: str) -> dict[str, int]:
    profile = normalize_for_match(researcher_profile)
    boosts = {"postdoc": 0, "phd": 0, "professor": 0}
    if not profile:
        return boosts

    if "r1" in profile or "first stage researcher" in profile or "doctoral" in profile:
        boosts["phd"] += 45
    if "r2" in profile or "r3" in profile or "recognised researcher" in profile:
        boosts["postdoc"] += 32
    if "established researcher" in profile:
        boosts["postdoc"] += 10
        boosts["professor"] += 10
    if "r4" in profile or "leading researcher" in profile:
        boosts["professor"] += 45
    return boosts


def classify_job_type(
    *,
    title: str | None,
    researcher_profile: str | None,
    cleaned_text: str | None,
    synonyms_path: str | None = None,
) -> JobTypeClassification:
    aliases = load_synonyms(synonyms_path)

    norm_title = normalize_for_match(title)
    norm_text = normalize_for_match(cleaned_text)
    scores = {"postdoc": 0, "phd": 0, "professor": 0}

    for job_type, variants in aliases.items():
        if any(_matches_alias(norm_title, alias) for alias in variants):
            scores[job_type] += TITLE_WEIGHT
        if any(_matches_alias(norm_text, alias) for alias in variants):
            scores[job_type] += TEXT_WEIGHT

    for job_type, boost in _profile_boosts(researcher_profile or "").items():
        scores[job_type] += boost

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score < MIN_SCORE or (best_score - second_score) < MIN_MARGIN:
        return JobTypeClassification(job_type="unknown", score=int(best_score), score_breakdown=scores)
    return JobTypeClassification(job_type=best_type, score=int(best_score), score_breakdown=scores)


def normalize_job_type_filter(value: str | None, *, synonyms_path: str | None = None) -> str | None:
    if value is None:
        return None
    normalized = normalize_for_match(value)
    if not normalized or normalized in {"all", "any"}:
        return None
    if normalized in {"postdoc", "phd", "professor"}:
        return normalized

    aliases = load_synonyms(synonyms_path)
    for job_type, variants in aliases.items():
        if normalized == job_type or normalized in variants:
            return job_type
    return None


def canonicalize_query(query: str, *, synonyms_path: str | None = None) -> str:
    normalized = normalize_for_match(query)
    if not normalized:
        return ""

    aliases = load_synonyms(synonyms_path)
    for job_type, variants in aliases.items():
        for alias in sorted(variants, key=len, reverse=True):
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            normalized = re.sub(pattern, job_type, normalized)
    return " ".join(normalized.split())


def expand_query(query: str, *, synonyms_path: str | None = None) -> str:
    canonical = canonicalize_query(query, synonyms_path=synonyms_path)
    if not canonical:
        return ""

    aliases = load_synonyms(synonyms_path)
    extras: list[str] = []
    tokens = set(canonical.split())
    for job_type, variants in aliases.items():
        if job_type not in tokens:
            continue
        extras.extend(alias for alias in variants if alias != job_type)

    merged = " ".join([canonical, *extras]).strip()
    return " ".join(merged.split())


def default_type_labels() -> dict[str, str]:
    return {
        "all": "All",
        "postdoc": "Postdoc",
        "phd": "PhD",
        "professor": "Professor",
    }

