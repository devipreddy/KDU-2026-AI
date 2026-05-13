# Setup Guide, Architecture Notes, And Demo Workflow

This guide shows how to run the local healthcare semantic search platform, exercise the ingestion
and retrieval pipeline, inspect audit logs, run benchmarks, and demo role-aware rendering.

All data in this repository is synthetic. Do not use real PHI.

## Current State

The backend implements the ingestion, extraction, PHI tokenization, chunking, indexing, retrieval,
rendering, audit, observability, timeline, and benchmark services as Python modules and FastAPI
endpoints where noted. The frontend currently demonstrates the search and audit experiences with
modeled local data while backend search APIs are still being wired in.

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop
- PowerShell on Windows

Optional but useful:

- Tesseract installed locally if you run the current OCR fallback worker
- A machine with enough memory for `sentence-transformers` models when running dense retrieval or
  cross-encoder reranking

## First-Time Setup

From the repository root:

```powershell
Copy-Item .env.example .env
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
```

Start local infrastructure:

```powershell
docker compose up -d postgres chromadb
```

Install backend dependencies and run migrations:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
alembic upgrade head
```

Install frontend dependencies:

```powershell
cd ..\frontend
npm install
```

Run services:

```powershell
# terminal 1
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload

# terminal 2
cd frontend
npm run dev
```

Useful local URLs:

- Frontend: `http://localhost:5173`
- Backend health: `http://localhost:8000/health`
- Backend config: `http://localhost:8000/api/v1/config`
- ChromaDB: `http://localhost:8001`

## Demo Users

Seeded users share the local password `ChangeMe123!`.

| Role | Email | Visibility |
| --- | --- | --- |
| Doctor | `doctor@example.com` | Full PHI when policy allows |
| Nurse | `nurse@example.com` | Limited clinical context |
| Admin | `admin@example.com` | Metadata-only |
| Researcher | `researcher@example.com` | De-identified clinical text |
| Records staff | `records@example.com` | Operational access |

Get a token:

```powershell
$login = Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/auth/login `
  -ContentType "application/json" `
  -Body '{"email":"doctor@example.com","password":"ChangeMe123!"}'

$token = $login.access_token
```

## Architecture Notes

The core backend flow is:

```text
upload
  -> locally encrypted document storage
  -> ingestion job
  -> typed extraction or OCR fallback
  -> OCR cleanup and medical normalization
  -> PHI detection and tokenization
  -> encrypted PHI mapping store
  -> clinical metadata extraction
  -> hierarchical parent/child chunking
  -> embedding generation
  -> ChromaDB indexing
  -> query understanding
  -> retrieval-time RBAC metadata filtering
  -> BM25 retrieval + dense retrieval
  -> reciprocal rank fusion
  -> cross-encoder reranking
  -> parent context expansion and citations
  -> patient timeline reconstruction
  -> role-aware PHI rendering
  -> immutable audit events and PHI-redacted observability
```

Storage surfaces:

- PostgreSQL stores users, roles, documents, chunks, ingestion jobs, PHI mappings, policies, and audit events.
- ChromaDB stores embedded chunks and scalar retrieval metadata.
- `data/storage` stores locally encrypted uploaded files.
- `data/processed` stores normalized/tokenized extracted text.
- `data/synthetic` stores the deterministic benchmark dataset.

Security posture:

- Authorization filters are built before retrieval so unauthorized chunks are not retrieved.
- Rendering enforces least-privilege PHI visibility after retrieval.
- Audit logging records rendered searches and per-document access.
- Logs and tracing payloads pass through PHI redaction before leaving the process.
- Local encryption is a development simulation, not production HIPAA compliance.

## Synthetic Dataset

The repository includes 1000 mock healthcare records and ground-truth labels:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.synthetic.generate_dataset
```

Files:

- `data/synthetic/records.jsonl`
- `data/synthetic/ground_truth_queries.json`
- `data/synthetic/manifest.json`

## Ingestion Demo

Create a text file from one synthetic record:

```powershell
$record = Get-Content .\data\synthetic\records.jsonl -First 1 | ConvertFrom-Json
$record.text | Set-Content -Encoding utf8 .\data\raw\demo-record.txt
```

Upload it as a doctor:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/documents/upload `
  -H "Authorization: Bearer $token" `
  -F "file=@data/raw/demo-record.txt;type=text/plain" `
  -F "patient_ref=$($record.patient_ref)" `
  -F "visit_id=$($record.visit_id)" `
  -F "document_type=clinical_note" `
  -F "hospital=$($record.hospital)" `
  -F "physician=$($record.physician)" `
  -F "department=$($record.department)" `
  -F "diagnosis=$($record.diagnosis)" `
  -F "icd_codes=$($record.icd_codes | ConvertTo-Json -Compress)" `
  -F "sensitivity_level=$($record.sensitivity_level)"
```

Process queued typed/text ingestion jobs:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.services.document_extraction --limit 10
```

For scanned PDFs or handwritten-note PDFs, run the OCR worker:

```powershell
.\.venv\Scripts\python.exe -m app.services.document_ocr --limit 10
```

The current local OCR worker is the initial Tesseract fallback. The architecture and synthetic
manifest remain aligned to the PaddleOCR target path.

## Indexing Demo

Inspect the ChromaDB collection mapping:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.services.chroma_index --print-mapping
```

Generate embeddings and upsert indexed chunks:

```powershell
.\.venv\Scripts\python.exe -m app.services.embedding_pipeline --limit 100
```

To index immediately after extraction in local demos, set this in `backend/.env` before starting
the backend:

```text
INDEX_ON_INGESTION=true
```

## Search Demo

Lightweight parser and authorization checks:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.services.query_understanding "patients with cardiac issues treated in Q1 2025"
.\.venv\Scripts\python.exe -m app.services.retrieval_authorization doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

Run retrieval stages:

```powershell
.\.venv\Scripts\python.exe -m app.services.bm25_retrieval doctor@example.com "I10 records by Dr. Asha Raman"
.\.venv\Scripts\python.exe -m app.services.dense_retrieval doctor@example.com "patients with cardiac issues treated in Q1 2025"
.\.venv\Scripts\python.exe -m app.services.hybrid_retrieval doctor@example.com "patients with cardiac issues treated in Q1 2025"
.\.venv\Scripts\python.exe -m app.services.cross_encoder_reranking doctor@example.com "patients with cardiac issues treated in Q1 2025" --rerank-top-n 20
.\.venv\Scripts\python.exe -m app.services.context_expansion doctor@example.com "patients with cardiac issues treated in Q1 2025"
.\.venv\Scripts\python.exe -m app.services.timeline_reconstruction doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

Dense retrieval and reranking require indexed chunks and may download local model weights.

## Role-Based Rendering Demo

Render the same query for different users:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.services.result_rendering doctor@example.com "patients with cardiac issues treated in Q1 2025"
.\.venv\Scripts\python.exe -m app.services.result_rendering researcher@example.com "patients with cardiac issues treated in Q1 2025"
.\.venv\Scripts\python.exe -m app.services.result_rendering admin@example.com "patients with cardiac issues treated in Q1 2025"
```

Expected behavior:

- Doctor: clinical text plus decrypted direct identifiers when policy allows.
- Researcher: clinical text with de-identified placeholders.
- Admin: metadata-only result records without clinical snippets.

The frontend also includes a visual role selector in the Search view so you can compare doctor,
researcher, and admin rendering behavior in the current UI prototype.

## Audit Log Demo

Rendered searches write audit events. After running `result_rendering`, inspect recent events:

```powershell
docker compose exec postgres psql `
  -U healthcare_app `
  -d healthcare_search `
  -c "select occurred_at, action, query_text, resource_type, result_document_ids, decision, role_snapshot from audit_events order by occurred_at desc limit 10;"
```

The frontend Audit Log tab demonstrates the target audit table: searchable events showing who
accessed what, when, which query was used, masking mode, and access decision.

## PHI Lookup Demo

Find a token from processed text:

```powershell
Get-Content .\data\processed\*.txt | Select-String "\[[A-Z0-9_]+\]" | Select-Object -First 5
```

Resolve it as an authorized doctor or records staff user:

```powershell
$tokenToResolve = "[REPLACE_WITH_TOKEN_FROM_PROCESSED_TEXT]"
$patientRef = "REPLACE_WITH_MATCHING_PATIENT_REF"

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/v1/phi/lookup `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" `
  -Body (@{ token = $tokenToResolve; patient_ref = $patientRef } | ConvertTo-Json)
```

Researchers and admins should receive a forbidden response for direct identifier lookup.

## Benchmark Demo

Run the benchmark suite:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.services.retrieval_quality_benchmarks
```

Metrics reported:

- top-3 accuracy
- wrong-patient retrieval rate
- average, p50, and p95 latency
- OCR success rate
- masking correctness

Fail the command when thresholds are not met:

```powershell
.\.venv\Scripts\python.exe -m app.services.retrieval_quality_benchmarks --fail-on-thresholds
```

For local oracle smoke testing, you can relax the OCR threshold:

```powershell
.\.venv\Scripts\python.exe -m app.services.retrieval_quality_benchmarks --ocr-success-min 0 --fail-on-thresholds
```

The current synthetic OCR confidence baseline is intentionally useful as a signal: it is below the
eventual 95% success target until OCR confidence/status improves.

## Observability Demo

Structured JSON logs are enabled by default and redact PHI fields. Optional tracing flags live in
`.env` and `backend/.env`:

```text
LOG_LEVEL=INFO
OTEL_ENABLED=false
OTEL_EXPORTER_OTLP_ENDPOINT=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=healthcare-semantic-search-local
```

Install optional observability dependencies only when exporting spans or LangSmith traces:

```powershell
cd backend
python -m pip install -e ".[observability]"
```

## Frontend Demo

Open:

```text
http://localhost:5173
```

Search view:

- Enter a natural language query.
- Change role between Doctor, Researcher, and Admin.
- Adjust document type, hospital, sensitivity, and confidence filters.
- Compare snippets, citations, confidence indicators, and metadata-only display.

Audit Log view:

- Search by user, query, document, patient ref, or IP.
- Filter by role, action, and decision.
- Review who accessed what, when, and with which query.

These frontend views are currently UI demos backed by local modeled data. The backend services above
are the source of truth for actual ingestion, retrieval, rendering, audit, and benchmark behavior.

## Useful Checks

Backend:

```powershell
cd backend
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m pytest
```

Frontend:

```powershell
cd frontend
npm run format:check
npm run typecheck
npm run lint
npm run build
```
