from __future__ import annotations

from euraxess_scraper.taxonomy import classify_job_type


def test_postdoc_variants_map_to_postdoc():
    hyphenated = classify_job_type(
        title="Post-doctoral Fellow in Computer Vision",
        researcher_profile="Recognised Researcher (R2)",
        cleaned_text="Research position in machine learning.",
    )
    compact = classify_job_type(
        title="Postdoc in Computational Neuroscience",
        researcher_profile="Recognised Researcher (R2)",
        cleaned_text="Research position in neuroscience.",
    )

    assert hyphenated.job_type == "postdoc"
    assert compact.job_type == "postdoc"


def test_professor_role_with_phd_requirement_stays_professor():
    classified = classify_job_type(
        title="Assistant Professor in Biomedical Engineering",
        researcher_profile="Leading Researcher (R4)",
        cleaned_text="Requirements include a PhD and strong publication record.",
    )
    assert classified.job_type == "professor"


def test_ambiguous_case_falls_back_to_unknown():
    classified = classify_job_type(
        title="Research position in life sciences",
        researcher_profile="",
        cleaned_text="Candidates should have a phd and postdoctoral experience.",
    )
    assert classified.job_type == "unknown"

