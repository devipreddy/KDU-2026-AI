# Healthcare Semantic Search Platform

Customer-replica learning project for a HIPAA-aware semantic search platform over synthetic medical records.

The first commit establishes the workspace foundation:

- FastAPI backend scaffold
- Vite React frontend scaffold
- Docker Compose for backend, frontend, PostgreSQL, and OpenSearch
- Shared environment templates
- Linting and formatting configuration
- Health endpoints for local smoke tests

## Architecture Direction

The system will be built feature by feature around these subsystems:

1. Secure document ingestion
2. OCR-aware extraction
3. PHI detection and tokenization
4. Medical metadata enrichment
5. Hierarchical chunking
6. Hybrid retrieval with OpenSearch
7. Cross-encoder reranking
8. Role-aware rendering
9. Immutable audit logging
10. Retrieval and OCR benchmarks

This repository currently contains only the scaffold needed for those future slices.

## Repository Layout

```text
.
├── backend/          FastAPI service
├── frontend/         Vite + React app
├── infra/            Local infrastructure config
├── data/             Local development data folders
├── docker-compose.yml
└── README.md
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop

## Local Environment

Create local environment files from the examples:

```powershell
Copy-Item .env.example .env
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
```

## Run With Docker Compose

```powershell
docker compose up --build
```

Useful URLs:

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Backend health: http://localhost:8000/health
- OpenSearch: http://localhost:9200

## Run Backend Locally

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Backend checks:

```powershell
ruff check .
ruff format --check .
pytest
```

## Run Frontend Locally

```powershell
cd frontend
npm install
npm run dev
```

Frontend checks:

```powershell
npm run lint
npm run format:check
npm run typecheck
```

## Security Posture For Local Development

This scaffold intentionally uses development defaults only. Real PHI should never be used in this project. Future commits will add synthetic data generation, role-aware access, PHI tokenization, encryption hooks, audit logs, and retrieval-time authorization.
