"""Zero-shot NLI-based job type classifier (optional / experimental).

Uses a HuggingFace NLI model (no API key, runs locally) to classify job postings
into postdoc / phd / professor / unknown.  Use this as a last-pass fallback on
the remaining ~770 jobs that the keyword+profile classifier leaves as "unknown".

Default model: cross-encoder/nli-deberta-v3-small (~90 MB, already cached).
Note: accuracy on academic job titles is moderate. The keyword+profile classifier
(reclassify command) handles 90%+ of cases more reliably; run that first.
"""
from __future__ import annotations

from functools import lru_cache

# Default model: small DeBERTa fine-tuned specifically for zero-shot classification.
# ~180 MB, fast on CPU, English. For multilingual corpora swap to:
#   "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  (~550 MB, covers FR/ES/DE/…)
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# Natural-language hypothesis labels → canonical job-type values.
# The pipeline tests each label via: "This job posting is for a <label>."
CANDIDATE_LABELS = [
    "postdoctoral researcher or research associate",
    "PhD student or doctoral candidate",
    "professor, lecturer, or faculty position",
]

LABEL_TO_TYPE: dict[str, str] = {
    "postdoctoral researcher or research associate": "postdoc",
    "PhD student or doctoral candidate": "phd",
    "professor, lecturer, or faculty position": "professor",
}

# Minimum NLI score to accept a classification; below this → "unknown"
MIN_CONFIDENCE = 0.50


@lru_cache(maxsize=4)
def _load_pipeline(model_name: str):
    """Load (and cache) the zero-shot-classification pipeline."""
    from transformers import pipeline  # type: ignore

    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,  # CPU; change to 0 for GPU
    )


def classify_job_type_nli(
    *,
    title: str | None,
    researcher_profile: str | None = None,
    cleaned_text: str | None = None,
    model_name: str = DEFAULT_NLI_MODEL,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[str, int]:
    """Classify a job posting into one of postdoc / phd / professor / unknown.

    Returns
    -------
    (job_type, score)
        job_type : "postdoc" | "phd" | "professor" | "unknown"
        score    : 0–100 integer (NLI confidence × 100), stored in job_type_score
    """
    parts: list[str] = []
    if title:
        parts.append(f"Job title: {title.strip()}")
    if researcher_profile:
        parts.append(f"Researcher profile: {researcher_profile.strip()}")
    if cleaned_text:
        # First ~500 chars of body text gives enough context without slowing inference
        parts.append(cleaned_text[:500].strip())

    if not parts:
        return "unknown", 0

    text = "\n".join(parts)

    try:
        pipe = _load_pipeline(model_name)
        result = pipe(
            text,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template="This job posting is for a {}.",
        )
        top_label: str = result["labels"][0]
        top_score: float = float(result["scores"][0])

        if top_score < min_confidence:
            return "unknown", int(top_score * 100)

        job_type = LABEL_TO_TYPE.get(top_label, "unknown")
        return job_type, int(top_score * 100)

    except Exception:
        return "unknown", 0
