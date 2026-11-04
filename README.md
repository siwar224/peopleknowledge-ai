# PeopleKnowledge AI

Base de projet pour une API RAG destinée à interroger des documents de politiques RH.

## Structure

```text
peopleknowledge-ai/
├── apps/api/
│   ├── app/
│   │   ├── main.py
│   │   └── rag/
│   └── pyproject.toml
├── datasets/
│   ├── source_documents/   # PDF à ajouter par le propriétaire du projet
│   ├── metadata/            # document_manifest.jsonl à ajouter
│   └── evaluation/          # rag_questions_gold.jsonl à ajouter
├── scripts/
├── docs/
├── infra/
├── .env.example
├── .gitignore
└── README.md
```

Les fichiers PDF et JSONL ne sont volontairement pas générés ici. Ajoutez-les ensuite dans les dossiers indiqués.

## Environnement Python isolé

Depuis la racine du projet :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".\apps\api[dev]"
```

Si PowerShell bloque l'activation, vous pouvez utiliser directement l'interpréteur de l'environnement :

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\apps\api[dev]"
```

Copiez ensuite `.env.example` vers `.env` et renseignez les secrets nécessaires :

```powershell
Copy-Item .env.example .env
```

## Lancer l'API

```powershell
python -m uvicorn app.main:app --app-dir apps/api --reload
```

Endpoint de contrôle : `http://127.0.0.1:8000/health`

Documentation interactive : `http://127.0.0.1:8000/docs`

## Vérifications développeur

```powershell
python -m ruff check apps/api
python -m pytest
```
