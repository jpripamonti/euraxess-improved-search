from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import config, db, search as search_mod
from ..taxonomy import default_type_labels

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
        country: str | None,
        page: int,
        page_size: int,
        active_only: bool,
        open_only: bool,
    ) -> dict:
        conn = db.get_connection(selected_db_path)
        try:
            db.init_db(conn)
            return search_mod.hybrid_search_page(
                conn,
                query=q,
                job_type=job_type,
                country=country,
                page=page,
                page_size=page_size,
                active_only=active_only,
                open_only=open_only,
                index_dir=selected_index_dir,
            )
        finally:
            conn.close()

    def _render_page(
        request: Request,
        *,
        q: str,
        job_type: str | None,
        country: str | None,
        page: int,
        page_size: int,
        active_only: bool,
        open_only: bool,
    ) -> HTMLResponse:
        payload = _search_payload(
            q=q,
            job_type=job_type,
            country=country,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
        )
        base_params = {
            "q": q,
            "job_type": payload["job_type"],
            "country": country or "",
            "page_size": page_size,
        }
        if active_only:
            base_params["active_only"] = "true"
        if open_only:
            base_params["open_only"] = "true"

        def _page_url(page_number: int) -> str:
            params = dict(base_params)
            params["page"] = page_number
            return f"{request.url_for('search_page')}?{urlencode(params)}"

        prev_page = (page - 1) if page > 1 else None
        next_page = (page + 1) if payload["has_next"] else None
        context = {
            "request": request,
            "payload": payload,
            "q": q,
            "country": country or "",
            "job_type": payload["job_type"],
            "active_only": active_only,
            "open_only": open_only,
            "type_labels": default_type_labels(),
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
            country=None,
            page=1,
            page_size=20,
            active_only=True,
            open_only=True,
        )

    @app.get("/search", response_class=HTMLResponse)
    def search_page(
        request: Request,
        q: str = Query("", description="Free-text search"),
        job_type: str | None = Query(None, description="all|postdoc|phd|professor"),
        country: str | None = Query(None),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        active_only: bool = Query(False),
        open_only: bool = Query(False),
    ) -> HTMLResponse:
        return _render_page(
            request,
            q=q,
            job_type=job_type,
            country=country,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
        )

    @app.get("/api/search")
    def api_search(
        q: str = Query("", description="Free-text search"),
        job_type: str | None = Query(None, description="all|postdoc|phd|professor"),
        country: str | None = Query(None),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        active_only: bool = Query(True),
        open_only: bool = Query(True),
    ) -> dict:
        return _search_payload(
            q=q,
            job_type=job_type,
            country=country,
            page=page,
            page_size=page_size,
            active_only=active_only,
            open_only=open_only,
        )

    return app
