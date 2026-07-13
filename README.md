# TorqBase

TorqBase is an AI-powered knowledge platform for technical two-stroke engine documentation. It lets users upload manuals, standards, reports, spreadsheets, images, and slide decks, then ask cited diagnostic questions and explore the extracted knowledge graph.

The project was built for a hackathon around the Hirth Engines challenge problem: making large, messy engineering document collections searchable, explainable, and useful for troubleshooting.

## Highlights

- Source-backed Q&A with citations and evidence snippets
- Multi-format ingestion for PDFs, DOCX, PPTX, spreadsheets, images, text, and URL records
- Hybrid retrieval over PostgreSQL, pgvector, lexical search, vector search, and reranking
- Knowledge Graph extraction for symptoms, causes, fixes, parts, specs, procedures, and source evidence
- Interactive 2D graph explorer with focus mode, ranked neighborhoods, filters, search, edge labels, path highlighting, and evidence inspection
- Corpus inventory workflow for safely scanning large engineering file trees before ingestion
- Formula/spec extraction helpers for technical values and tabular data
- Dark, polished static frontend served by FastAPI

## Demo Flow

1. Upload or index technical documents.
2. Ask an engineering question, for example: "What can cause engine misfire?"
3. Inspect the cited answer and evidence snippets.
4. Open the Knowledge Graph to explore related symptoms, causes, fixes, parts, and source-backed relationships.
5. Use focus mode to reduce graph clutter and explain the diagnostic path step by step.

## Tech Stack

- Backend: FastAPI, Python
- Frontend: static HTML/CSS/JavaScript served by FastAPI
- Retrieval: PostgreSQL, pgvector, hybrid lexical/vector search, reranking
- AI orchestration: agentic retrieval pipeline, verifier, local or OpenAI-compatible LLM endpoint
- Knowledge graph: rule-assisted extraction, normalized ontology, interactive D3/SVG visualization
- Infrastructure: Docker Compose for local PostgreSQL + pgvector

## Repository Layout

```text
.
|-- twostroke-kb/
|   |-- api/                 # FastAPI app and static frontend
|   |-- agent/               # Retrieval, answer generation, verifier, KG retrieval
|   |-- ingestion/           # Parsers, chunking, dedup, KG extraction, inventory
|   |-- db/                  # PostgreSQL schema
|   |-- docs/                # Architecture notes
|   |-- scripts/             # Maintenance/backfill scripts
|   `-- tests/               # Smoke tests and fixtures
|-- Hirth_Architecture.md
|-- Hirth_Challenge_Plan.md
`-- System Architecture.md
```

## Quick Start

Run these commands from `twostroke-kb/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

## Configuration

Copy `twostroke-kb/.env.example` to `twostroke-kb/.env` and configure:

- `DATABASE_URL`: PostgreSQL connection string
- `LLM_PROVIDER`: `local`, `openai`, or another OpenAI-compatible provider
- `LLM_BASE_URL`: local Ollama/vLLM/OpenAI-compatible endpoint
- `LLM_MODEL`: model name used for answer generation

For local development, the default database expects Docker Compose to expose PostgreSQL on port `5433`.

## Notes For Reviewers

- Built as a hackathon project for exploring AI-assisted technical documentation, diagnostics, and knowledge graph navigation.
- The Knowledge Graph is intentionally kept as a stable 2D D3/SVG explorer. An experimental 3D graph view was removed because it did not improve usability.
- Local LLM response time depends heavily on the machine and model. For public deployment, a hosted OpenAI-compatible model is recommended.
- Uploaded corpora and generated local data are intentionally ignored by Git.

## License

No license has been selected yet. Treat this repository as source-available for review unless a license is added.
