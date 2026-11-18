# PeopleKnowledge AI

Project base for a RAG API that answers questions about HR policy documents.

## Structure

```text
peopleknowledge-ai/
├── pyproject.toml              # dependencies + ruff/pytest config
├── src/peopleknowledge/        # the Python package, one folder per RAG phase
│   ├── config.py
│   ├── ingestion/              # OFFLINE: documents -> parsed documents -> chunks
│   ├── retrieval/              # question -> relevant chunks (bge-m3 + Qdrant)
│   ├── generation/             # chunks -> grounded answer with trusted citations
│   ├── evaluation/             # gold questions + retrieval metrics
│   └── api/                    # FastAPI endpoints + test page
├── scripts/                    # command-line runners of the pipeline
├── tests/                      # same tree as src/peopleknowledge/
├── datasets/
│   ├── source_documents/       # the corpus (kept local)
│   ├── metadata/               # company profile, rules, document catalog
│   └── evaluation/             # rag_questions_gold.jsonl
└── docs/
```

## Isolated Python environment

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks the activation, you can call the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Then copy `.env.example` to `.env` and fill in the required secrets:

```powershell
Copy-Item .env.example .env
```

## Run the API

```powershell
python -m uvicorn peopleknowledge.api.main:app --reload
```

Health check endpoint: `http://127.0.0.1:8000/health`

Interactive documentation: `http://127.0.0.1:8000/docs`

## Developer checks

```powershell
python -m ruff check .
python -m pytest
```
