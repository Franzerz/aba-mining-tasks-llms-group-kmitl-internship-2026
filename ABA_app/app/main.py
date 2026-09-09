#### PATCHARAKORN ####

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import DIST_DIR
from .routes.graph import router as graph_router
from .routes.review import router as review_router

app = FastAPI(title="ABA App")

# API routers first, so /api/* is never swallowed by the SPA catch-all below.
app.include_router(graph_router)
app.include_router(review_router)

if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/")
def index() -> FileResponse:
    return _spa_index()


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    """Client-side routing means /review and /graph are real URLs to the browser but not to
    the server — serve index.html for any non-API path so a refresh or a direct link works.
    A genuinely missing static file (e.g. /favicon.ico) still 404s properly."""
    # An unmatched /api/* path must 404, not fall through to index.html — otherwise a typo'd
    # or removed endpoint returns 200 with HTML and the caller only fails later on res.json().
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"No such API endpoint: /{full_path}")

    candidate = (DIST_DIR / full_path).resolve()
    if candidate.is_file() and candidate.is_relative_to(DIST_DIR.resolve()):
        return FileResponse(candidate)
    if "." in full_path.rsplit("/", 1)[-1]:  # looks like a file request, not a route
        raise HTTPException(status_code=404, detail=f"Not found: {full_path}")
    return _spa_index()


def _spa_index() -> FileResponse:
    index_html = DIST_DIR / "index.html"
    if not index_html.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend not built yet. Run: cd frontend && npm install && npm run build",
        )
    return FileResponse(index_html)
