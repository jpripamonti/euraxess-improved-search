from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import config, db, search as search_mod
from ..taxonomy import default_type_labels
from ..topics import TOPIC_OTHER, default_topic_labels

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

    def _search_payload(
        *,
        q: str,
        job_type: str | None,
        topic: str | None,
        include_topics: list[str] | None,
        exclude_topics: list[str] | None,
        country: str | None,
        page: int,
        page_size: int,
        active_only: bool,
        open_only: bool,
        debug: bool,
    ) -> dict:
        conn = db.get_connection(selected_db_path)
        try:
            db.init_db(conn)
            return search_mod.hybrid_search_page(
                conn,
                query=q,
                job_type=job_type,
                topic_domain=topic,
                topic_domains=include_topics,
                exclude_topic_domains=exclude_topics,
                country=country,
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
        topic: str | None,
        include_topics: list[str] | None,
        exclude_topics: list[str] | None,
        country: str | None,
        page: int,
        page_size: int,
        active_only: bool,
        open_only: bool,
        debug: bool,
    ) -> HTMLResponse:
        payload = _search_payload(
            q=q,
            job_type=job_type,
            topic=topic,
            include_topics=include_topics,
            exclude_topics=exclude_topics,
            country=country,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
            debug=debug,
        )
        base_params: dict[str, Any] = {
            "q": q,
            "job_type": payload["job_type"],
            "country": country or "",
            "page_size": page_size,
        }
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
        context = {
            "request": request,
            "payload": payload,
            "q": q,
            "country": country or "",
            "job_type": payload["job_type"],
            "topic": payload["topic"],
            "include_topics": payload.get("include_topics") or [],
            "exclude_topics": payload.get("exclude_topics") or [],
            "active_only": active_only,
            "open_only": open_only,
            "debug": debug,
            "type_labels": default_type_labels(),
            "topic_labels": default_topic_labels(),
            "topic_filter_labels": {k: v for k, v in default_topic_labels().items() if k != "all"},
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
            topic=None,
            include_topics=[],
            exclude_topics=[],
            country=None,
            page=1,
            page_size=20,
            active_only=True,
            open_only=True,
            debug=False,
        )

    @app.get("/search", response_class=HTMLResponse)
    def search_page(
        request: Request,
        q: str = Query("", description="Free-text search"),
        job_type: str | None = Query(None, description="all|postdoc|phd|professor"),
        topic: str | None = Query(None, description="legacy single-topic filter"),
        include_topic: list[str] | None = Query(None, description="Include any of these topics"),
        exclude_topic: list[str] | None = Query(None, description="Exclude these topics"),
        country: str | None = Query(None),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        active_only: bool = Query(True),
        open_only: bool = Query(True),
        debug: bool = Query(False),
    ) -> HTMLResponse:
        return _render_page(
            request,
            q=q,
            job_type=job_type,
            topic=topic,
            include_topics=include_topic or [],
            exclude_topics=exclude_topic or [],
            country=country,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
            debug=debug,
        )

    @app.get("/api/search")
    def api_search(
        q: str = Query("", description="Free-text search"),
        job_type: str | None = Query(None, description="all|postdoc|phd|professor"),
        topic: str | None = Query(None, description="legacy single-topic filter"),
        include_topic: list[str] | None = Query(None, description="Include any of these topics"),
        exclude_topic: list[str] | None = Query(None, description="Exclude these topics"),
        country: str | None = Query(None),
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
            db.init_db(conn)
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
            db.init_db(conn)
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
