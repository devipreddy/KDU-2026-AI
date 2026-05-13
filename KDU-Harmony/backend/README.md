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
.\.venv\Scripts\python.exe -m app.services.embedding_pipeline --retry-failed
```

The first embedding worker uses `BAAI/bge-base-en-v1.5` through sentence-transformers. It stores
the Chroma embedding ID and collection on each `document_chunks` row and includes embedding model
metadata in the Chroma payload.
Each chunk also tracks `pending`, `indexing`, `indexed`, or `failed` status with attempt counts,
timestamps, and failure messages. Set `INDEX_ON_INGESTION=true` to index freshly extracted chunks
as part of the ingestion worker; otherwise the embedding worker can be run as an explicit retryable
indexing step.

## Query Understanding

Parse natural-language retrieval queries into metadata-aware signals:

```powershell
.\.venv\Scripts\python.exe -m app.services.query_understanding "patients with cardiac issues treated in Q1 2025"
```

The parser extracts temporal filters, diagnosis concepts, patient references, hospitals, physicians,
document types, and ICD codes so the retrieval layer can build Chroma metadata filters before search.

## Retrieval Authorization Filters

Build Chroma metadata filters with RBAC constraints applied before retrieval:

```powershell
.\.venv\Scripts\python.exe -m app.services.retrieval_authorization doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

The filter builder combines parsed query signals with active document access policies, especially
allowed sensitivity levels, so unauthorized chunks are excluded before vector or keyword retrieval.

## BM25 Retrieval

Run the local lexical retrieval path over authorized chunks:

```powershell
.\.venv\Scripts\python.exe -m app.services.bm25_retrieval doctor@example.com "I10 records by Dr. Asha Raman"
```

BM25 retrieval scores exact keyword matches across medication names, ICD codes, authorized MRN
tokens, physician names, and diagnosis terms after RBAC metadata filters have narrowed the candidate
set.

## Dense Vector Retrieval

Run semantic retrieval over embedded chunks stored in ChromaDB:

```powershell
.\.venv\Scripts\python.exe -m app.services.dense_retrieval doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

Dense retrieval embeds the parsed query, sends ChromaDB the same authorization-aware metadata
filters used by BM25, and hydrates returned Chroma chunk IDs back to database-backed snippets.

## Hybrid Retrieval

Combine BM25 and dense vector candidates with reciprocal rank fusion:

```powershell
.\.venv\Scripts\python.exe -m app.services.hybrid_retrieval doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

Hybrid retrieval runs both retrievers with the same parsed query and RBAC scope, then ranks the
union of chunk candidates with RRF so records found by both methods rise above single-source hits.

## Cross-Encoder Reranking

Rerank the top hybrid candidates with a local cross-encoder:

```powershell
.\.venv\Scripts\python.exe -m app.services.cross_encoder_reranking doctor@example.com "patients with cardiac issues treated in Q1 2025" --rerank-top-n 20
```

The reranker uses `BAAI/bge-reranker-large` by default and falls back to
`BAAI/bge-reranker-base` if the larger model cannot be loaded. Only the top-N hybrid candidates
are scored by the cross-encoder to keep latency bounded.

## Parent Context And Citations

Return citation-ready results with matched chunks expanded to their parent clinical section:

```powershell
.\.venv\Scripts\python.exe -m app.services.context_expansion doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

Context expansion returns the matched chunk, parent section text, source document, page number,
section, citation label, retrieval scores, and a confidence score that incorporates reranker,
hybrid, source, and OCR confidence signals.

## PHI-Aware Rendering

Render final search results according to role-based PHI visibility:

```powershell
.\.venv\Scripts\python.exe -m app.services.result_rendering doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

Doctors and records staff receive decrypted direct identifiers when their policies allow it,
researchers receive de-identified clinical text, and administrators receive metadata-only results
without clinical snippets or parent section text.

## Patient Timeline Reconstruction

Reconstruct patient timelines from contextual retrieval results:

```powershell
.\.venv\Scripts\python.exe -m app.services.timeline_reconstruction doctor@example.com "patients with cardiac issues treated in Q1 2025"
```

Timeline reconstruction groups matched chunks by patient reference, visit date, hospital,
diagnosis, and document type, then preserves source citations, chunk IDs, sections, visit IDs,
ICD codes, first rank, and highest confidence for each grouped visit/document slice.

## Observability

The backend emits structured JSON logs with PHI redaction enabled by default. Request logs include
method, path, status, duration, and a request ID, while the redaction processor removes direct
identifiers, PHI tokens, emails, phone numbers, MRNs, SSNs, DOB strings, and sensitive structured
fields before log records are rendered.

Optional OpenTelemetry and LangSmith hooks are configured through environment variables:

```powershell
LOG_LEVEL=INFO
OTEL_ENABLED=false
OTEL_EXPORTER_OTLP_ENDPOINT=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=healthcare-semantic-search-local
```

Install optional tracing packages when exporting spans or LangSmith traces:

```powershell
python -m pip install -e ".[observability]"
```

LangSmith trace payloads are prepared through the same PHI redaction layer before being attached
to tracing context.

## Retrieval Audit Logging

PHI-aware rendering writes append-only audit events for every rendered search. Each retrieval logs
the user, roles, query, authorization/query filters, returned document IDs, matched chunk IDs,
timestamp, masking mode, PHI visibility, and access decision. A separate document access audit event
is also written for each source document returned in the result set.

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
