# TorqBase Application

This directory contains the runnable TorqBase app: a FastAPI backend, static frontend, ingestion pipelines, retrieval/agent code, PostgreSQL schema, and tests.

For the project overview, see the repository-level `README.md`.

## What The App Does

- Serves a polished landing page and main diagnostic workspace.
- Ingests technical documents into chunks with metadata and citations.
- Supports PDF, DOCX, PPTX, spreadsheets, images/OCR, text, and `.url` link records.
- Answers questions with source-backed citations.
- Builds and explores a diagnostic Knowledge Graph.
- Provides graph filters, search, focus mode, ranked neighborhoods, path highlighting, and evidence inspection.
- Includes a corpus inventory workflow for scanning large file trees safely before ingestion.

## Local Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install system tools used by parsers where needed:

- Tesseract OCR
- Poppler
- LibreOffice

Ubuntu example:

```bash
sudo apt install tesseract-ocr tesseract-ocr-deu poppler-utils libreoffice
```

macOS example:

```bash
brew install tesseract tesseract-lang poppler libreoffice
```

## Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Important variables:

- `DATABASE_URL`: PostgreSQL/pgvector database URL
- `LLM_PROVIDER`: local or hosted provider
- `LLM_BASE_URL`: OpenAI-compatible API base URL
- `LLM_MODEL`: answer-generation model
- `EMBEDDING_MODEL`: embedding model
- `UPLOAD_DIR`: local upload directory

Local Docker Compose defaults to:

```text
postgresql://postgres:postgres@localhost:5433/twostroke
```

## Database

Start PostgreSQL with pgvector:

```bash
docker compose up -d
```

The schema in `db/schema.sql` is mounted into the container and is applied on first database initialization.

## Run

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Useful Checks

```bash
python -m py_compile config.py api/main.py
pytest
```

Smoke-test the graph endpoint after the database is running:

```bash
curl "http://127.0.0.1:8000/graph?limit=1200&min_confidence=0.4&include_seed=true"
```

## Implementation Notes

- The frontend is intentionally static HTML/CSS/JavaScript. There is no React/Vite/Tailwind build step.
- The Knowledge Graph view uses D3/SVG and remains the stable default exploration surface.
- Local uploaded files, generated images, logs, caches, and `.env` files are ignored by Git.
- Local LLMs can be slow on CPU-bound machines. For public deployment, configure a hosted OpenAI-compatible endpoint.

## Main Modules

- `api/main.py`: FastAPI routes, static serving, upload/chat/graph endpoints
- `api/static/index.html`: main frontend app
- `api/static/landing.html`: landing page
- `ingestion/`: parsers, chunking, inventory, KG extraction
- `agent/`: retrieval, answer generation, verifier, KG retrieval
- `memory/`: conversation persistence
- `db/schema.sql`: database schema
- `tests/`: smoke tests and fixtures
