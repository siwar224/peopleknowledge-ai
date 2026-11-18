# PeopleKnowledge API

Web layer of PeopleKnowledge AI (FastAPI). It only wires HTTP to the RAG phases in `ingestion/`, `retrieval/`, `generation/` and `evaluation/`.

Main endpoints:

- `POST /api/search`: retrieve ranked chunks;
- `POST /api/answer`: retrieve evidence, build the context and prompt, call the local model, validate citations, and return the answer;
- `GET /api/gold`: load the evaluation questions;
- `GET /api/profile-options`: load stable profile identifiers for the web interface.

The web interface at `/` displays the generated answer, trusted sources, gold-question comparison, and retrieval diagnostics.
