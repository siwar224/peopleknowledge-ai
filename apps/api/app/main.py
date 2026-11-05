"""Minimal entry point of the PeopleKnowledge API."""

from fastapi import FastAPI

app = FastAPI(title="PeopleKnowledge API", version="0.1.0")


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Check that the API responds."""

    return {"status": "ok"}
