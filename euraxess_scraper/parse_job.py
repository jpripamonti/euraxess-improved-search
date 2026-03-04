from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup, Tag

from .language import detect_language
from .taxonomy import classify_job_type
from .utils import clean_text, extract_job_id, parse_date_to_utc_iso, sha256_text

SECTION_IDS = {
    "offer_description": "offer-description",
    "where_to_apply": "where-to-apply",
    "requirements": "requirements",
    "additional_information": "additional-information",
    "work_locations": "work-locations",
    "contact": "contact",
}


def _section_text(container: Tag, heading_text: str) -> str:
    text = container.get_text("\n", strip=True)
    text = re.sub(r"\n+", "\n", text)
    if text.startswith(heading_text):
        text = text[len(heading_text) :].strip()
    return clean_text(text)


def _extract_sections(soup: BeautifulSoup) -> dict[str, str]:
    sections: dict[str, str] = {}
    for key, section_id in SECTION_IDS.items():
        heading = soup.find(id=section_id)
        if not heading:
            sections[key] = ""
            continue
        container = heading.find_parent("div")
        if container is None:
            sections[key] = ""
            continue
        heading_text = clean_text(heading.get_text(" ", strip=True))
        sections[key] = _section_text(container, heading_text)
    return sections


def _parse_dt_dd_map(container: Tag | None) -> dict[str, str]:
    if container is None:
        return {}
    mapping: dict[str, str] = {}
    for dl in container.find_all("dl"):
        terms = dl.find_all("dt")
        defs = dl.find_all("dd")
        for term, definition in zip(terms, defs):
            key = clean_text(term.get_text(" ", strip=True)).lower()
            value = clean_text(definition.get_text(" ", strip=True))
            if key:
                mapping[key] = value
    return mapping


def _extract_city_from_text(value: str) -> str | None:
    match = re.search(r"city\s+(.+?)(?:\s+website\s+|\s+street\s+|\s+postal\s+code\s+|$)", value, re.I)
    if match:
        return clean_text(match.group(1))
    return None


def parse_job_detail(html: str, url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    job_id = extract_job_id(url)

    og_title = soup.find("meta", attrs={"property": "og:title"})
    page_title = clean_text(og_title.get("content")) if og_title else ""

    h1 = soup.find("h1")
    h1_text = clean_text(h1.get_text(" ", strip=True)) if h1 else ""

    title = page_title or h1_text
    if title.lower() == "job offer":
        strong = soup.select_one("#offer-description strong")
        if strong:
            title = clean_text(strong.get_text(" ", strip=True))

    job_info_heading = soup.find(id="job-information")
    job_info_container = job_info_heading.find_parent("div") if job_info_heading else None
    info_map = _parse_dt_dd_map(job_info_container)

    posted_date_iso = None
    meta_time = soup.select_one(".ecl-content-item__meta time[datetime]")
    if meta_time:
        posted_date_iso = parse_date_to_utc_iso(meta_time.get("datetime"))
    if not posted_date_iso:
        meta_text = soup.select_one(".ecl-content-item__meta")
        if meta_text:
            posted_date_iso = parse_date_to_utc_iso(meta_text.get_text(" ", strip=True))

    deadline_iso = None
    deadline_time = None
    if job_info_container:
        deadline_time = job_info_container.select_one("time[datetime]")
    if deadline_time:
        deadline_iso = parse_date_to_utc_iso(deadline_time.get("datetime"))
    if not deadline_iso:
        deadline_iso = parse_date_to_utc_iso(info_map.get("application deadline"))

    sections = _extract_sections(soup)
    work_locations_text = sections.get("work_locations", "")

    work_loc_heading = soup.find(id="work-locations")
    work_loc_container = work_loc_heading.find_parent("div") if work_loc_heading else None
    work_map = _parse_dt_dd_map(work_loc_container)

    city = work_map.get("city") or _extract_city_from_text(work_locations_text)

    cleaned_chunks = [v for v in sections.values() if v]
    cleaned_text = clean_text("\n\n".join(cleaned_chunks))
    inferred = classify_job_type(
        title=title,
        researcher_profile=info_map.get("researcher profile"),
        cleaned_text=cleaned_text,
    )

    position_type = "job_offer"
    salary = info_map.get("salary") or info_map.get("offered salary")

    record = {
        "job_id": job_id,
        "url": url,
        "title": title,
        "organization": info_map.get("organisation/company"),
        "country": info_map.get("country") or work_map.get("country"),
        "city": city,
        "posted_date": posted_date_iso,
        "deadline": deadline_iso,
        "researcher_profile": info_map.get("researcher profile"),
        "position_type": position_type,
        "job_type_inferred": inferred.job_type,
        "job_type_score": inferred.score,
        "contract_type": info_map.get("type of contract"),
        "hours": info_map.get("job status") or info_map.get("hours per week"),
        "salary": salary,
        "cleaned_text": cleaned_text,
        "sections_json": json.dumps(sections, ensure_ascii=False, sort_keys=True),
        "content_hash": sha256_text(cleaned_text) if cleaned_text else None,
        "language": detect_language(cleaned_text) if cleaned_text else None,
    }
    return record
