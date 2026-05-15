# Healthcare Semantic Search Platform

Customer-replica learning project for a HIPAA-aware semantic search platform over synthetic medical records.

For the full local setup, architecture notes, and demo workflow, see
[docs/setup-demo.md](docs/setup-demo.md).

The workspace includes:

- FastAPI backend service
- Vite React frontend app
- Docker Compose for backend, frontend, PostgreSQL, and ChromaDB
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
6. Hybrid retrieval with ChromaDB and keyword indexes
7. Cross-encoder reranking
8. Role-aware rendering
9. Immutable audit logging
10. Retrieval and OCR benchmarks

The current implementation includes feature slices for ingestion, tokenization, chunking, ChromaDB
indexing, hybrid retrieval, role-aware rendering, audit logging, observability, timelines, and
benchmarks. The frontend search view now calls the live backend pipeline, including BM25, dense
retrieval, RRF, cross-encoder reranking, parent context expansion, PHI-aware rendering, audit
logging, and optional OpenAI Responses API answer generation.

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

Then follow the guided workflow in [docs/setup-demo.md](docs/setup-demo.md).

## Run With Docker Compose

```powershell
docker compose up --build
```

Useful URLs:

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Backend health: http://localhost:8000/health
- ChromaDB: http://localhost:8001

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

Database migrations:

```powershell
cd backend
alembic upgrade head
```

## Synthetic Dataset

The repository includes a deterministic synthetic dataset for ingestion and retrieval benchmarks:

- `data/synthetic/records.jsonl`: 1000 fabricated healthcare records
- `data/synthetic/ground_truth_queries.json`: query labels and expected record IDs
- `data/synthetic/manifest.json`: dataset counts and generation metadata

Regenerate it with:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.synthetic.generate_dataset
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

This local project intentionally uses development defaults only. Real PHI should never be used in
this project. The implemented feature slices use synthetic data generation, role-aware access, PHI
tokenization, local encryption hooks, audit logs, retrieval-time authorization, and an OpenRouter
Qianfan OCR worker for scanned and handwritten PDFs.

Seeded local auth users use the shared demo password `ChangeMe123!` and are intended only for local development.

Uploaded documents are written to local encrypted storage under `data/storage` by default. The
local encryption mode is a development simulation and should not be used for real PHI.

Typed PDF and plain text extraction writes normalized extracted text under `data/processed`.
Scanned PDF OCR uses OpenRouter's `baidu/qianfan-ocr-fast` model and surfaces provider errors
directly in the ingestion dashboard.
Multi-page OCR can run page requests in parallel with `OPENROUTER_OCR_MAX_WORKERS`; lower it to `1`
for strict rate limits or raise it cautiously for faster multi-page ingestion.
Both typed extraction and OCR pass through medical text normalization before processed text is stored.
Direct identifiers found during extraction are tokenized before processed text is written, with encrypted
lookup mappings stored in the database for role-aware rendering.
Authorized doctors and records staff can resolve tokens through an audited PHI lookup endpoint; researchers,
administrators, and limited-access users remain blocked from decrypting direct identifiers.
Ingestion also extracts clinical metadata such as diagnoses, medications, symptoms, ICD codes,
physicians, hospitals, dates, and section offsets for later metadata-aware retrieval.
Processed records are chunked hierarchically: section-sized parent chunks preserve clinical context,
while smaller child chunks support precise retrieval and later ChromaDB embedding.
The ChromaDB collection schema now defines dense-vector storage, BM25 lexical fields, patient/document
metadata filters, and scalar Chroma metadata used by the upcoming hybrid retrieval path.
Chunk embeddings are generated with `BAAI/bge-base-en-v1.5` through sentence-transformers and stored
in ChromaDB alongside the retrieval metadata.
Chunk indexing now has durable status tracking, retryable failure capture, and an optional
`INDEX_ON_INGESTION=true` hook that indexes freshly extracted chunks directly into ChromaDB.
Natural-language queries are parsed into temporal, diagnosis, patient, hospital, physician,
document-type, and ICD-code signals before the upcoming retrieval step builds metadata filters.
Retrieval authorization now combines those query signals with active RBAC policies so ChromaDB
receives pre-filtered metadata constraints and unauthorized chunks stay out of candidate retrieval.
Clinician organization scopes are enforced before BM25 and dense retrieval, and doctors only receive
decrypted patient identifiers when they have an active patient-care assignment.
BM25 lexical retrieval now scores authorized chunks for exact medication names, ICD codes, MRN
tokens, physician names, and diagnosis terms.
Dense vector retrieval now embeds natural-language queries and searches authorized ChromaDB chunks
semantically before hydrating snippets from the database.
Hybrid retrieval now combines BM25 and dense vector rankings with reciprocal rank fusion.
Cross-encoder reranking now reorders the top hybrid candidates with a BGE reranker and smaller
fallback model.
Parent context expansion now returns matched chunks with parent section context, source citations,
page/section metadata, and confidence signals.
PHI-aware rendering now decrypts assigned clinician views, de-identifies researcher and unassigned
doctor results, and returns metadata-only administrator views.
Patient timeline reconstruction now groups retrieved results by patient, visit date, hospital,
diagnosis, and document type while preserving citations and confidence signals.
Observability now emits PHI-redacted structured JSON logs and includes optional OpenTelemetry and
LangSmith tracing hooks.
Retrieval latency is reduced with reused Chroma clients/collections, single-flight cached embedding
and reranker model loads, and a cached Presidio analyzer for ingestion PHI detection.
Retrieval quality benchmarks now measure top-3 accuracy, wrong-patient retrieval rate, latency,
OCR success rate, and masking correctness against the synthetic ground-truth suite.
Retrieval audit logging now records rendered searches and per-document access with query filters,
document/chunk IDs, masking mode, timestamp, roles, and access decision.
The frontend now includes a live role-aware natural-language search workspace with filters,
generated answers, retrieval-stage status, snippets, citations, confidence indicators, and
metadata-only rendering.
The frontend also includes a live searchable audit log table showing who accessed which records,
when, with which query, masking mode, and access decision.
