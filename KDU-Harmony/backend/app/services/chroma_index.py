from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from app.core.config import settings
from app.models.document_chunk import DocumentChunk

HYBRID_INDEX_SCHEMA_VERSION = "chroma_hybrid_medical_chunks_v1"
DENSE_VECTOR_FIELD = "embedding"
DOCUMENT_TEXT_FIELD = "content"

BM25_FIELDS = (
    "content",
    "section",
    "diagnosis",
    "medications",
    "symptoms",
    "icd_codes",
    "physician",
    "hospital",
    "document_type",
)

METADATA_FILTER_FIELDS = (
    "patient_ref",
    "visit_id",
    "document_id",
    "external_id",
    "chunk_id",
    "parent_chunk_id",
    "chunk_type",
    "section",
    "document_type",
    "hospital",
    "physician",
    "diagnosis",
    "icd_codes",
    "sensitivity_level",
    "visit_date",
    "ocr_confidence",
)

PATIENT_DOCUMENT_FIELDS = (
    "patient_ref",
    "visit_id",
    "document_id",
    "external_id",
    "document_type",
    "chunk_id",
    "parent_chunk_id",
)

SCALAR_METADATA_TYPES = (str, int, float, bool)


class ChromaIndexError(RuntimeError):
    """Raised when Chroma index setup cannot be completed."""


@dataclass(frozen=True)
class HybridIndexMapping:
    collection_name: str
    schema_version: str
    dense_vector_field: str
    document_text_field: str
    bm25_fields: tuple[str, ...]
    metadata_filter_fields: tuple[str, ...]
    patient_document_fields: tuple[str, ...]
    distance_metric: str = "cosine"

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "schema_version": self.schema_version,
            "dense_vector_field": self.dense_vector_field,
            "document_text_field": self.document_text_field,
            "bm25_fields": list(self.bm25_fields),
            "metadata_filter_fields": list(self.metadata_filter_fields),
            "patient_document_fields": list(self.patient_document_fields),
            "distance_metric": self.distance_metric,
        }

    def to_collection_metadata(self) -> dict[str, str]:
        return {
            "hnsw:space": self.distance_metric,
            "schema_version": self.schema_version,
            "dense_vector_field": self.dense_vector_field,
            "document_text_field": self.document_text_field,
            "bm25_fields": json.dumps(list(self.bm25_fields)),
            "metadata_filter_fields": json.dumps(list(self.metadata_filter_fields)),
            "patient_document_fields": json.dumps(list(self.patient_document_fields)),
        }


@dataclass(frozen=True)
class ChromaChunkRecord:
    id: str
    document: str
    metadata: dict[str, str | int | float | bool]


def hybrid_index_mapping(collection_name: str | None = None) -> HybridIndexMapping:
    return HybridIndexMapping(
        collection_name=collection_name or settings.chroma_collection,
        schema_version=HYBRID_INDEX_SCHEMA_VERSION,
        dense_vector_field=DENSE_VECTOR_FIELD,
        document_text_field=DOCUMENT_TEXT_FIELD,
        bm25_fields=BM25_FIELDS,
        metadata_filter_fields=METADATA_FILTER_FIELDS,
        patient_document_fields=PATIENT_DOCUMENT_FIELDS,
    )


def build_chroma_record(chunk: DocumentChunk) -> ChromaChunkRecord:
    return ChromaChunkRecord(
        id=chroma_id_for_chunk(chunk),
        document=build_bm25_document(chunk),
        metadata=build_chroma_metadata(chunk),
    )


def build_chroma_upsert_payload(
    chunks: list[DocumentChunk],
    *,
    embeddings: list[list[float]] | None = None,
) -> dict[str, Any]:
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError("Embedding count must match chunk count")

    records = [build_chroma_record(chunk) for chunk in chunks]
    payload: dict[str, Any] = {
        "ids": [record.id for record in records],
        "documents": [record.document for record in records],
        "metadatas": [record.metadata for record in records],
    }
    if embeddings is not None:
        payload["embeddings"] = embeddings
    return payload


def build_chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
    retrieval_metadata = chunk.retrieval_metadata or {}
    clinical_entities = retrieval_metadata.get("clinical_entities") or {}
    embedding_metadata = retrieval_metadata.get("embedding") or {}
    dates = clinical_entities.get("dates") if isinstance(clinical_entities, dict) else []
    visit_date = first_date_value(dates)

    metadata: dict[str, Any] = {
        "schema_version": HYBRID_INDEX_SCHEMA_VERSION,
        "chunk_id": str(chunk.id),
        "parent_chunk_id": str(chunk.parent_chunk_id) if chunk.parent_chunk_id else "",
        "chunk_index": chunk.chunk_index,
        "chunk_type": retrieval_metadata.get(
            "chunk_type", "child" if chunk.parent_chunk_id else "parent"
        ),
        "section": chunk.section or retrieval_metadata.get("section", ""),
        "patient_ref": retrieval_metadata.get("patient_ref", ""),
        "visit_id": retrieval_metadata.get("visit_id", ""),
        "document_id": retrieval_metadata.get("document_id", str(chunk.document_id)),
        "external_id": retrieval_metadata.get("external_id", ""),
        "document_type": retrieval_metadata.get("document_type", ""),
        "hospital": retrieval_metadata.get("hospital", ""),
        "physician": retrieval_metadata.get("physician", ""),
        "diagnosis": retrieval_metadata.get("diagnosis", ""),
        "icd_codes": join_values(retrieval_metadata.get("icd_codes")),
        "sensitivity_level": retrieval_metadata.get(
            "sensitivity_level", chunk.sensitivity_level.value
        ),
        "visit_date": visit_date,
        "token_count": chunk.token_count,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "ocr_confidence": float(chunk.ocr_confidence) if chunk.ocr_confidence is not None else None,
        "embedding_model": (
            embedding_metadata.get("model") if isinstance(embedding_metadata, dict) else None
        ),
        "embedding_dimension": (
            embedding_metadata.get("dimension") if isinstance(embedding_metadata, dict) else None
        ),
        "embedding_indexed_at": (
            embedding_metadata.get("indexed_at") if isinstance(embedding_metadata, dict) else None
        ),
        "content_sha256": chunk.content_sha256,
        "bm25_content": chunk.content,
        "bm25_section": chunk.section or "",
        "bm25_diagnosis": retrieval_metadata.get("diagnosis", ""),
        "bm25_medications": join_values(
            clinical_entities.get("medications") if isinstance(clinical_entities, dict) else None
        ),
        "bm25_symptoms": join_values(
            clinical_entities.get("symptoms") if isinstance(clinical_entities, dict) else None
        ),
        "bm25_icd_codes": join_values(
            clinical_entities.get("icd_codes") if isinstance(clinical_entities, dict) else None
        ),
        "bm25_physician": retrieval_metadata.get("physician", ""),
        "bm25_hospital": retrieval_metadata.get("hospital", ""),
        "bm25_document_type": retrieval_metadata.get("document_type", ""),
    }
    return flatten_chroma_metadata(metadata)


def build_bm25_document(chunk: DocumentChunk) -> str:
    retrieval_metadata = chunk.retrieval_metadata or {}
    clinical_entities = retrieval_metadata.get("clinical_entities") or {}
    lexical_parts = [
        chunk.content,
        chunk.section or "",
        retrieval_metadata.get("diagnosis", ""),
        retrieval_metadata.get("physician", ""),
        retrieval_metadata.get("hospital", ""),
        retrieval_metadata.get("document_type", ""),
    ]
    if isinstance(clinical_entities, dict):
        lexical_parts.extend(
            [
                join_values(clinical_entities.get("diagnoses")),
                join_values(clinical_entities.get("medications")),
                join_values(clinical_entities.get("symptoms")),
                join_values(clinical_entities.get("icd_codes")),
            ]
        )
    return "\n".join(part for part in lexical_parts if part)


def flatten_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        normalized = normalize_chroma_value(value)
        if normalized is None:
            continue
        flattened[key] = normalized
    return flattened


def normalize_chroma_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, SCALAR_METADATA_TYPES):
        return value
    if isinstance(value, list | tuple | set):
        return join_values(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def join_values(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    try:
        return "|".join(str(item) for item in value if item is not None)
    except TypeError:
        return str(value)


def first_date_value(dates: Any) -> str:
    if not isinstance(dates, list):
        return ""
    for date_entry in dates:
        if isinstance(date_entry, dict) and date_entry.get("value"):
            return str(date_entry["value"])
    return ""


def chroma_id_for_chunk(chunk: DocumentChunk) -> str:
    return chunk.embedding_id or f"chunk:{chunk.id}"


def get_chroma_client() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise ChromaIndexError(
            "chromadb is not installed. Install backend dependencies before bootstrapping Chroma."
        ) from exc

    if settings.chroma_host:
        parsed_url = urlparse(str(settings.chroma_host))
        if not parsed_url.hostname:
            raise ChromaIndexError("CHROMA_HOST must include a hostname")
        return chromadb.HttpClient(
            host=parsed_url.hostname,
            port=parsed_url.port or (443 if parsed_url.scheme == "https" else 80),
            ssl=parsed_url.scheme == "https",
        )

    return chromadb.PersistentClient(path=str(settings.chroma_persist_path.resolve()))


def ensure_chroma_collection(
    client: Any | None = None, *, collection_name: str | None = None
) -> Any:
    mapping = hybrid_index_mapping(collection_name=collection_name)
    chroma_client = client or get_chroma_client()
    return chroma_client.get_or_create_collection(
        name=mapping.collection_name,
        metadata=mapping.to_collection_metadata(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the ChromaDB hybrid chunk collection.")
    parser.add_argument("--collection", default=settings.chroma_collection)
    parser.add_argument("--print-mapping", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = hybrid_index_mapping(collection_name=args.collection)
    if args.print_mapping:
        print(json.dumps(mapping.to_dict(), indent=2, sort_keys=True))
        return

    collection = ensure_chroma_collection(collection_name=args.collection)
    print(
        json.dumps(
            {
                "collection": collection.name,
                "schema_version": HYBRID_INDEX_SCHEMA_VERSION,
                "dense_vector_field": DENSE_VECTOR_FIELD,
                "bm25_fields": list(BM25_FIELDS),
                "metadata_filter_fields": list(METADATA_FILTER_FIELDS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
