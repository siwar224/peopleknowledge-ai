# PeopleKnowledge AI

Project base for a RAG API that answers questions about HR policy documents.

## Structure

```text
peopleknowledge-ai/
├── apps/api/
│   ├── app/
│   │   ├── main.py
│   │   └── rag/
│   └── pyproject.toml
├── datasets/
│   ├── source_documents/   # PDFs to be added by the project owner
│   ├── metadata/            # document_manifest.jsonl to be added
│   └── evaluation/          # rag_questions_gold.jsonl to be added
├── scripts/
├── docs/
├── infra/
├── .env.example
├── .gitignore
└── README.md
```

The PDF and JSONL files are intentionally not generated here. Add them afterwards to the folders shown above.

## Isolated Python environment

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".\apps\api[dev]"
```

If PowerShell blocks the activation, you can call the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\apps\api[dev]"
```

Then copy `.env.example` to `.env` and fill in the required secrets:

```powershell
Copy-Item .env.example .env
```

## Run the API

```powershell
python -m uvicorn app.main:app --app-dir apps/api --reload
```

Health check endpoint: `http://127.0.0.1:8000/health`

Interactive documentation: `http://127.0.0.1:8000/docs`

## Developer checks

```powershell
python -m ruff check apps/api
python -m pytest
```
