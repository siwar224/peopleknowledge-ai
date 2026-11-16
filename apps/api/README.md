# PeopleKnowledge API

Python package of the PeopleKnowledge API. The application code lives in `app/`, and the RAG components in `app/rag/`.

Main endpoints:

- `POST /api/search`: retrieve ranked chunks;
- `POST /api/answer`: retrieve evidence, build the context and prompt, call the local model, validate citations, and return the answer;
- `GET /api/gold`: load the evaluation questions;
- `GET /api/profile-options`: load stable profile identifiers for the web interface.

The web interface at `/` displays the generated answer, trusted sources, gold-question comparison, and retrieval diagnostics.
