# Backend

FastAPI service for the healthcare semantic search platform.

This scaffold exposes:

- `GET /health`
- `GET /api/v1/health`
- `GET /api/v1/config`

The first migration adds core tables for users, roles, documents, chunks, ingestion jobs, PHI mappings, audit events, and access policies.

Run migrations from this directory:

```powershell
alembic upgrade head
```

Feature commits will add authentication, ingestion, PaddleOCR extraction, PHI handling, ChromaDB retrieval, audit logging, and benchmarks.

## Authentication

Local development includes seeded demo users after migrations:

| Role | Email | Password |
| --- | --- | --- |
| Doctor | `doctor@example.com` | `ChangeMe123!` |
| Nurse | `nurse@example.com` | `ChangeMe123!` |
| Admin | `admin@example.com` | `ChangeMe123!` |
| Researcher | `researcher@example.com` | `ChangeMe123!` |
| Records Staff | `records@example.com` | `ChangeMe123!` |

Login endpoint:

```text
POST /api/v1/auth/login
```

Current user endpoint:

```text
GET /api/v1/auth/me
```

## Document Upload

Authenticated doctors, administrators, and records staff can upload PDF or plain text records:

```text
POST /api/v1/documents/upload
```

The endpoint validates file type and content signature, computes a SHA-256 checksum, writes a locally encrypted storage object, creates document metadata, and queues an ingestion job.

## Text Extraction

Process queued typed PDF and plain text ingestion jobs:

```powershell
.\.venv\Scripts\python.exe -m app.services.document_extraction
```

The pipeline decrypts the local storage object, verifies the original SHA-256 checksum, extracts UTF-8 text or typed PDF text with PyMuPDF, writes processed text under `../data/processed`, and marks the ingestion job as succeeded. OCR-required documents remain queued for the PaddleOCR slice.

Extracted text is normalized before storage: whitespace is collapsed, headings are canonicalized,
paragraphs are restored, OCR artifacts are removed, and common medical OCR misspellings are corrected.
Names, DOBs, MRNs, phone numbers, addresses, and emails are then replaced with stable PHI tokens
before processed text is written. The encrypted token-to-value mappings are stored in `phi_mappings`.
The same ingestion pass extracts clinical metadata for diagnoses, medications, symptoms, ICD codes,
physicians, hospitals, dates, and document section offsets. High-confidence fields backfill document
metadata columns when the upload did not provide them.
It then creates hierarchical chunks in `document_chunks`: parent chunks map to clinical sections and
child chunks split those sections into smaller overlapping retrieval units.

## ChromaDB Index Mapping

Bootstrap or inspect the ChromaDB hybrid retrieval collection:

```powershell
.\.venv\Scripts\python.exe -m app.services.chroma_index --print-mapping
.\.venv\Scripts\python.exe -m app.services.chroma_index
```

The mapping defines the dense vector field, BM25 lexical fields, scalar metadata filters, and
patient/document identifiers used by retrieval-time authorization.

## Embedding Generation

Generate embeddings for unindexed chunks and upsert them into ChromaDB:

```powershell
.\.venv\Scripts\python.exe -m app.services.embedding_pipeline --limit 100
```

The first embedding worker uses `BAAI/bge-base-en-v1.5` through sentence-transformers. It stores
the Chroma embedding ID and collection on each `document_chunks` row and includes embedding model
metadata in the Chroma payload.

## PHI Lookup

Authorized users can resolve a stored token through:

```text
POST /api/v1/phi/lookup
```

The endpoint decrypts values from the separate `phi_mappings` store only when the user's RBAC policy
allows PHI visibility. Each allowed, denied, or not-found lookup writes a `phi_decrypt` audit event.

## OCR Fallback

Process queued scanned PDF ingestion jobs with the initial Tesseract OCR fallback:

```powershell
.\.venv\Scripts\python.exe -m app.services.document_ocr
```

The OCR worker renders PDF pages, extracts text with Tesseract, stores average OCR confidence, writes processed text, and routes low-confidence documents to `review_required`.

## Synthetic Records

Generate the local synthetic healthcare dataset:

```powershell
.\.venv\Scripts\python.exe -m app.synthetic.generate_dataset
```

The generator writes `records.jsonl`, `manifest.json`, and `ground_truth_queries.json` to `../data/synthetic`.
