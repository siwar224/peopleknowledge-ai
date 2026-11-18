"""Entry point of the PeopleKnowledge API.

Start it from the project root:
    python -m uvicorn peopleknowledge.api.main:app --port 8000
then open http://localhost:8000 for the search test page.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .answer import router as answer_router
from .search import router as search_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="PeopleKnowledge API", version="0.1.0")
app.include_router(search_router)
app.include_router(answer_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Check that the API responds."""

    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def search_page() -> FileResponse:
    """Assistant page: ask a question, get an answer, and inspect its evidence."""

    return FileResponse(STATIC_DIR / "search.html", media_type="text/html")
