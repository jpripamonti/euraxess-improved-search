from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import config, db, search as search_mod
from ..language import language_label
from ..taxonomy import default_type_labels
from ..topics import TOPIC_OTHER, default_topic_labels, normalize_topic_filters

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(
    *,
    db_path: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="EURAXESS Search",
        description="Web UI for hybrid EURAXESS search with strict type filters.",
        version="0.1.0",
    )
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    selected_db_path = Path(db_path) if db_path is not None else config.DB_PATH
    selected_index_dir = Path(index_dir) if index_dir is not None else config.INDEX_DIR

    # Run schema setup once at startup instead of on every request.
    _init_conn = db.get_connection(selected_db_path)
    try:
        db.init_db(_init_conn)
    finally:
        _init_conn.close()

    def _build_facet_labels(
        db_path: Path,
        *,
        country: str | None,
        language: str | None,
        job_type: str | None,
        include_topics: list[str] | None,
        exclude_topics: list[str] | None,
        active_only: bool,
        open_only: bool,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
        """Return labels for country/language/type/topic filters with context-aware counts."""
        conn = db.get_connection(db_path)
        try:
            facet_kwargs: dict[str, Any] = dict(
                country=country,
                language=language,
                job_type=job_type,
                include_topics=include_topics or [],
                exclude_topics=exclude_topics or [],
                active_only=active_only,
                open_only=open_only,
            )
            country_counts = db.facet_counts(conn, "country", **facet_kwargs)
            lang_counts = db.facet_counts(conn, "language", **facet_kwargs)
            type_counts = db.facet_counts(conn, "job_type_inferred", include_unknown=True, **facet_kwargs)
            topic_counts = db.facet_counts(conn, "topic_domain", **facet_kwargs)
        finally:
            conn.close()

        country_labels: dict[str, str] = {"all": "All countries"}
        country_name_labels: dict[str, str] = {}
        for name, count in country_counts:
            country_labels[name] = f"{name} ({count:,})"
            country_name_labels[name] = name

        lang_labels: dict[str, str] = {"all": "All languages"}
        lang_name_labels: dict[str, str] = {}  # plain names without counts, for display in cards
        for code, count in lang_counts:
            name = language_label(code)
            lang_labels[code] = f"{name} ({count:,})"
            lang_name_labels[code] = name

        type_count_map = dict(type_counts)
        PRIMARY_TYPE_KEYS = ("all", "postdoc", "phd", "professor")
        DROPDOWN_LABELS: dict[str, str] = {
            "all": "All",
            "postdoc": "Postdoc",
            "phd": "PhD",
            "professor": "Professor",
            "other": "Research Support / Technical Staff",
            "unknown": "Open Call / Any Level",
        }
        primary_type_labels: dict[str, str] = {}
        for key in PRIMARY_TYPE_KEYS:
            label = DROPDOWN_LABELS[key]
            if key == "all":
                primary_type_labels[key] = label
            elif key in type_count_map:
                primary_type_labels[key] = f"{label} ({type_count_map[key]:,})"
            else:
                primary_type_labels[key] = label

        extended_type_labels: dict[str, str] = {"": "None"}
        for key in ("other", "unknown"):
            if key in type_count_map:
                extended_type_labels[key] = f"{DROPDOWN_LABELS[key]} ({type_count_map[key]:,})"

        topic_count_map = dict(topic_counts)
        base_topic = default_topic_labels()
        topic_filter_labels: dict[str, str] = {}
        for key, label in base_topic.items():
            if key == "all":
                continue
            if key in topic_count_map:
                topic_filter_labels[key] = f"{label} ({topic_count_map[key]:,})"
            else:
                topic_filter_labels[key] = label

        summary_type_labels = {key: value for key, value in DROPDOWN_LABELS.items() if key != "all"}
        summary_type_labels["all"] = "All"

        return (
            country_labels,
            country_name_labels,
            lang_labels,
            lang_name_labels,
            primary_type_labels,
            extended_type_labels,
            topic_filter_labels,
            summary_type_labels,
        )

    def _search_payload(
        *,
        q: str,
        job_type: str | None,
        topic: str | None,
        include_topics: list[str] | None,
        exclude_topics: list[str] | None,
        country: str | None,
        language: str | None,
        page: int,
        page_size: int,
        active_only: bool,
        open_only: bool,
        debug: bool,
    ) -> dict:
        conn = db.get_connection(selected_db_path)
        try:
            return search_mod.hybrid_search_page(
                conn,
                query=q,
                job_type=job_type,
                topic_domain=topic,
                topic_domains=include_topics,
                exclude_topic_domains=exclude_topics,
                country=country,
                language=language,
                page=page,
                page_size=page_size,
                active_only=active_only,
                open_only=open_only,
                debug=debug,
                index_dir=selected_index_dir,
            )
        finally:
            conn.close()

    def _render_page(
        request: Request,
        *,
        q: str,
        job_type: str | None,
        extended_job_type: str | None,
        topic: str | None,
        include_topics: list[str] | None,
        exclude_topics: list[str] | None,
        country: str | None,
        language: str | None,
        page: int,
        page_size: int,
        active_only: bool,
        open_only: bool,
        debug: bool,
        min_role_confidence: int,
        min_topic_confidence: int,
    ) -> HTMLResponse:
        effective_job_type = extended_job_type or job_type
        effective_country = None if country in {None, "", "all"} else country
        raw_include_topics = normalize_topic_filters(include_topics or [])
        raw_exclude_topics = normalize_topic_filters(exclude_topics or [])
        topic_overlap = [value for value in raw_include_topics if value in set(raw_exclude_topics)]

        payload = _search_payload(
            q=q,
            job_type=effective_job_type,
            topic=topic,
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            country=effective_country,
            language=language,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
            debug=debug,
        )
        base_params: dict[str, Any] = {
            "q": q,
            "country": country or "all",
            "page_size": page_size,
        }
        if payload["job_type"] in {"other", "unknown"}:
            base_params["extended_job_type"] = payload["job_type"]
        else:
            base_params["job_type"] = payload["job_type"]
        if min_role_confidence > 0:
            base_params["min_role_confidence"] = min_role_confidence
        if min_topic_confidence > 0:
            base_params["min_topic_confidence"] = min_topic_confidence
        if payload.get("language") and payload["language"] != "all":
            base_params["language"] = payload["language"]
        for value in payload.get("include_topics") or []:
            base_params.setdefault("include_topic", []).append(value)
        for value in payload.get("exclude_topics") or []:
            base_params.setdefault("exclude_topic", []).append(value)
        if active_only:
            base_params["active_only"] = "true"
        if open_only:
            base_params["open_only"] = "true"
        if debug:
            base_params["debug"] = "true"

        current_url = request.url.path
        if request.url.query:
            current_url = f"{current_url}?{request.url.query}"
        for row in payload["results"]:
            back_encoded = urlencode({"back": current_url})
            row["detail_url"] = f"{request.url_for('job_detail_page', job_id=row['job_id'])}?{back_encoded}"

        def _page_url(page_number: int) -> str:
            params = dict(base_params)
            params["page"] = page_number
            return f"{request.url_for('search_page')}?{urlencode(params, doseq=True)}"

        prev_page = (page - 1) if page > 1 else None
        next_page = (page + 1) if payload["has_next"] else None
        (
            country_labels,
            country_name_labels,
            lang_labels,
            lang_name_labels,
            primary_type_labels,
            extended_type_labels,
            topic_filter_labels,
            summary_type_labels,
        ) = _build_facet_labels(
            selected_db_path,
            country=effective_country,
            language=payload.get("language") if payload.get("language") != "all" else None,
            job_type=payload["job_type"] if payload["job_type"] != "all" else None,
            include_topics=payload.get("include_topics") or [],
            exclude_topics=payload.get("exclude_topics") or [],
            active_only=active_only,
            open_only=open_only,
        )
        context = {
            "request": request,
            "payload": payload,
            "q": q,
            "country": effective_country or "all",
            "job_type": payload["job_type"] if payload["job_type"] in {"all", "postdoc", "phd", "professor"} else "all",
            "extended_job_type": payload["job_type"] if payload["job_type"] in {"other", "unknown"} else "",
            "topic": payload["topic"],
            "include_topics": payload.get("include_topics") or [],
            "exclude_topics": payload.get("exclude_topics") or [],
            "topic_overlap_labels": [default_topic_labels().get(value, value) for value in topic_overlap],
            "language": payload.get("language") or "all",
            "active_only": active_only,
            "open_only": open_only,
            "debug": debug,
            "min_role_confidence": min_role_confidence,
            "min_topic_confidence": min_topic_confidence,
            "country_labels": country_labels,
            "country_name_labels": country_name_labels,
            "primary_type_labels": primary_type_labels,
            "extended_type_labels": extended_type_labels,
            "summary_type_labels": summary_type_labels,
            "type_name_labels": default_type_labels(),  # short labels for card pills (no counts)
            "topic_labels": default_topic_labels(),
            "topic_filter_labels": topic_filter_labels,
            "language_labels": lang_labels,
            "language_name_labels": lang_name_labels,
            "prev_page": prev_page,
            "next_page": next_page,
            "prev_url": _page_url(prev_page) if prev_page else None,
            "next_url": _page_url(next_page) if next_page else None,
        }
        return templates.TemplateResponse(request, "index.html", context)

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return _render_page(
            request,
            q="",
            job_type=None,
            extended_job_type=None,
            topic=None,
            include_topics=[],
            exclude_topics=[],
            country=None,
            language=None,
            page=1,
            page_size=20,
            active_only=True,
            open_only=True,
            debug=False,
            min_role_confidence=0,
            min_topic_confidence=0,
        )

    @app.get("/search", response_class=HTMLResponse)
    def search_page(
        request: Request,
        q: str = Query("", description="Free-text search"),
        job_type: str | None = Query(None, description="all|postdoc|phd|professor"),
        extended_job_type: str | None = Query(None, description="other|unknown"),
        topic: str | None = Query(None, description="legacy single-topic filter"),
        include_topic: list[str] | None = Query(None, description="Include any of these topics"),
        exclude_topic: list[str] | None = Query(None, description="Exclude these topics"),
        country: str | None = Query(None),
        language: str | None = Query(None, description="ISO 639-1 language code, e.g. en, de, fr"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        active_only: bool = Query(True),
        open_only: bool = Query(True),
        debug: bool = Query(False),
        min_role_confidence: int = Query(0, ge=0, le=100),
        min_topic_confidence: int = Query(0, ge=0, le=100),
    ) -> HTMLResponse:
        return _render_page(
            request,
            q=q,
            job_type=job_type,
            extended_job_type=extended_job_type,
            topic=topic,
            include_topics=include_topic or [],
            exclude_topics=exclude_topic or [],
            country=country,
            language=language,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
            debug=debug,
            min_role_confidence=min_role_confidence,
            min_topic_confidence=min_topic_confidence,
        )

    @app.get("/api/search")
    def api_search(
        q: str = Query("", description="Free-text search"),
        job_type: str | None = Query(None, description="all|postdoc|phd|professor"),
        topic: str | None = Query(None, description="legacy single-topic filter"),
        include_topic: list[str] | None = Query(None, description="Include any of these topics"),
        exclude_topic: list[str] | None = Query(None, description="Exclude these topics"),
        country: str | None = Query(None),
        language: str | None = Query(None, description="ISO 639-1 language code, e.g. en, de, fr"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        active_only: bool = Query(True),
        open_only: bool = Query(True),
        debug: bool = Query(False),
    ) -> dict:
        return _search_payload(
            q=q,
            job_type=job_type,
            topic=topic,
            include_topics=include_topic or [],
            exclude_topics=exclude_topic or [],
            country=country,
            language=language,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
            debug=debug,
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse, name="job_detail_page")
    def job_detail_page(
        request: Request,
        job_id: str,
        back: str | None = Query(None, description="Back URL to search page"),
    ) -> HTMLResponse:
        conn = db.get_connection(selected_db_path)
        try:
            row = db.get_job_detail(conn, job_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            job = dict(row)
            sections = db.parse_sections_json(row)
            context = {
                "request": request,
                "job": job,
                "sections": sections,
                "topic_labels": default_topic_labels(),
                "topic_domain": job.get("topic_domain") or TOPIC_OTHER,
                "back_url": back or str(request.url_for("home")),
            }
            return templates.TemplateResponse(request, "job_detail.html", context)
        finally:
            conn.close()

    @app.get("/api/jobs/{job_id}")
    def api_job_detail(job_id: str) -> dict:
        conn = db.get_connection(selected_db_path)
        try:
            row = db.get_job_detail(conn, job_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            payload = dict(row)
            payload["sections"] = db.parse_sections_json(row)
            payload["topic_domain"] = payload.get("topic_domain") or TOPIC_OTHER
            return payload
        finally:
            conn.close()

    return app
