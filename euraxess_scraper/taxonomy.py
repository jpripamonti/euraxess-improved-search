from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

CANONICAL_JOB_TYPES: tuple[str, ...] = ("postdoc", "phd", "professor", "other", "unknown")

TITLE_WEIGHT = 70
TEXT_WEIGHT = 12
MIN_SCORE = 25   # lowered from 35 so R2-only profile boost (45) clears the bar
MIN_MARGIN = 10  # lowered slightly for multi-keyword edge cases

# Score assigned when we use the researcher_profile as a direct (exclusive) signal
_PROFILE_DIRECT_SCORE = 80


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


def _profile_direct(researcher_profile: str) -> str | None:
    """Return a definitive job type when the researcher profile is unambiguous.

    Returns None when the profile is missing, mixed (all levels listed), or
    ambiguous, in which case normal keyword scoring should decide.
    """
    profile = normalize_for_match(researcher_profile)
    if not profile:
        return None

    has_r1 = bool(re.search(r"\br1\b", profile)) or "first stage researcher" in profile
    has_r2 = bool(re.search(r"\br2\b", profile)) or "recognised researcher" in profile
    has_r3 = bool(re.search(r"\br3\b", profile)) or "established researcher" in profile
    has_r4 = bool(re.search(r"\br4\b", profile)) or "leading researcher" in profile
    has_other = "other profession" in profile

    academic = [has_r1, has_r2, has_r3, has_r4]
    n_academic = sum(academic)

    # Non-academic with no academic profile → "other"
    if has_other and n_academic == 0:
        return "other"

    # All four levels listed → open call, don't use profile as signal
    if n_academic >= 3:
        return None

    # Exclusive R1 → PhD student
    if has_r1 and not has_r2 and not has_r3 and not has_r4:
        return "phd"

    # Exclusive R2 (or R2+R3 which is the postdoc range) → postdoc
    if (has_r2 or has_r3) and not has_r1 and not has_r4:
        return "postdoc"

    # Exclusive R4 → professor / group leader
    if has_r4 and not has_r1 and not has_r2 and not has_r3:
        return "professor"

    # R3+R4 → senior, lean professor
    if has_r3 and has_r4 and not has_r1 and not has_r2:
        return "professor"

    return None


def _profile_boosts(researcher_profile: str) -> dict[str, int]:
    """Return score boosts for non-exclusive profiles (used as soft signal)."""
    profile = normalize_for_match(researcher_profile)
    boosts = {"postdoc": 0, "phd": 0, "professor": 0}
    if not profile:
        return boosts

    if bool(re.search(r"\br1\b", profile)) or "first stage researcher" in profile:
        boosts["phd"] += 45
    if bool(re.search(r"\br2\b", profile)) or "recognised researcher" in profile:
        boosts["postdoc"] += 45
    if "established researcher" in profile:
        boosts["postdoc"] += 20
        boosts["professor"] += 15
    if bool(re.search(r"\br4\b", profile)) or "leading researcher" in profile:
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

    # 1. Title keywords — highest priority; a clear title match always wins over profile
    title_hits: dict[str, int] = {}
    for job_type, variants in aliases.items():
        if any(_matches_alias(norm_title, alias) for alias in variants):
            title_hits[job_type] = TITLE_WEIGHT

    if title_hits:
        # Build full scores (title + text + soft boosts) for the winning type
        scores = dict(title_hits)
        for job_type, variants in aliases.items():
            if any(_matches_alias(norm_text, alias) for alias in variants):
                scores[job_type] = scores.get(job_type, 0) + TEXT_WEIGHT
        for job_type, boost in _profile_boosts(researcher_profile or "").items():
            scores[job_type] = scores.get(job_type, 0) + boost

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1]
        # Accept if the title-matched type wins with sufficient margin
        if best_type in title_hits and (best_score - second_score) >= MIN_MARGIN:
            return JobTypeClassification(job_type=best_type, score=int(best_score), score_breakdown=scores)

    # 2. No unambiguous title keyword → try exclusive researcher_profile signal
    direct = _profile_direct(researcher_profile or "")
    if direct is not None:
        scores = {"postdoc": 0, "phd": 0, "professor": 0}
        if direct != "other":
            scores[direct] = _PROFILE_DIRECT_SCORE
        return JobTypeClassification(
            job_type=direct,
            score=_PROFILE_DIRECT_SCORE,
            score_breakdown=scores,
        )

    # 3. Fallback: combined title + text keywords + soft profile boosts
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
    if normalized in {"postdoc", "phd", "professor", "other", "unknown"}:
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
    """Labels for each job type.

    Used both for the UI dropdown (with counts appended in app.py) and as short
    display labels in result cards.  Keep the "other" and "unknown" labels concise
    enough to work as card pills.
    """
    return {
        "all": "All",
        "postdoc": "Postdoc",
        "phd": "PhD",
        "professor": "Professor",
        # "other" = jobs classified via Other Profession researcher profile
        #            (lab technicians, software engineers, research support staff)
        "other": "Support Staff",
        # "unknown" = genuinely ambiguous open calls (all-4-profiles, no clear title keyword)
        "unknown": "Open Call",
    }

