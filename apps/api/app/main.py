"""Point d'entrée minimal de l'API PeopleKnowledge."""

from fastapi import FastAPI

app = FastAPI(title="PeopleKnowledge API", version="0.1.0")


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Vérifie que l'API répond."""

    return {"status": "ok"}
