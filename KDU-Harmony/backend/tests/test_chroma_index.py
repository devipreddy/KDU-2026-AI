import hashlib
from uuid import UUID

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentStatus, DocumentType, SensitivityLevel
from app.services.chroma_index import (
    BM25_FIELDS,
    DENSE_VECTOR_FIELD,
    HYBRID_INDEX_SCHEMA_VERSION,
    METADATA_FILTER_FIELDS,
    build_chroma_record,
    build_chroma_upsert_payload,
    ensure_chroma_collection,
    hybrid_index_mapping,
)


class FakeChromaClient:
    def __init__(self) -> None:
        self.created_collection_name: str | None = None
        self.created_collection_metadata: dict | None = None

    def get_or_create_collection(self, *, name: str, metadata: dict):
        self.created_collection_name = name
        self.created_collection_metadata = metadata
        return FakeChromaCollection(name=name, metadata=metadata)


class FakeChromaCollection:
    def __init__(self, *, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = metadata


def build_test_chunk() -> DocumentChunk:
    document_id = UUID("40000000-0000-4000-8000-000000000001")
    chunk_id = UUID("50000000-0000-4000-8000-000000000001")
    parent_chunk_id = UUID("50000000-0000-4000-8000-000000000000")
    content = "Medications: metoprolol 25 mg BID for hypertension."
    document = Document(
        id=document_id,
        external_id="DOC-TEST-0001",
        patient_ref="PATIENT_REF_0042",
        visit_id="VISIT-0042",
        document_type=DocumentType.CLINICAL_NOTE,
        status=DocumentStatus.PROCESSED,
        file_name="note.txt",
        source_uri="local-encrypted://note",
        mime_type="text/plain",
        checksum_sha256="a" * 64,
        hospital="Harmony General Hospital",
        physician="Dr. Asha Raman",
        diagnosis="Hypertension",
        icd_codes=["I10"],
        sensitivity_level=SensitivityLevel.HIGH,
        is_encrypted=True,
        ocr_required=False,
        document_metadata={},
    )
    return DocumentChunk(
        id=chunk_id,
        document=document,
        document_id=document_id,
        parent_chunk_id=parent_chunk_id,
        chunk_index=3,
        section="Medications",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        token_count=7,
        start_offset=150,
        end_offset=200,
        sensitivity_level=SensitivityLevel.HIGH,
        retrieval_metadata={
            "chunk_type": "child",
            "section": "Medications",
            "patient_ref": "PATIENT_REF_0042",
            "visit_id": "VISIT-0042",
            "document_id": str(document_id),
            "external_id": "DOC-TEST-0001",
            "document_type": "clinical_note",
            "hospital": "Harmony General Hospital",
            "physician": "Dr. Asha Raman",
            "diagnosis": "Hypertension",
            "icd_codes": ["I10"],
            "sensitivity_level": "high",
            "clinical_entities": {
                "diagnoses": ["Hypertension"],
                "medications": ["metoprolol 25 mg BID"],
                "symptoms": ["chest pressure"],
                "icd_codes": ["I10"],
                "dates": [{"label": "visit_date", "value": "2025-02-14"}],
            },
        },
    )


def test_hybrid_index_mapping_declares_dense_bm25_and_filter_fields() -> None:
    mapping = hybrid_index_mapping(collection_name="unit_test_chunks")
    collection_metadata = mapping.to_collection_metadata()

    assert mapping.collection_name == "unit_test_chunks"
    assert mapping.schema_version == HYBRID_INDEX_SCHEMA_VERSION
    assert mapping.dense_vector_field == DENSE_VECTOR_FIELD
    assert "content" in BM25_FIELDS
    assert "medications" in BM25_FIELDS
    assert "patient_ref" in METADATA_FILTER_FIELDS
    assert "document_id" in METADATA_FILTER_FIELDS
    assert collection_metadata["hnsw:space"] == "cosine"
    assert collection_metadata["dense_vector_field"] == "embedding"
    assert "patient_ref" in collection_metadata["metadata_filter_fields"]


def test_build_chroma_record_flattens_metadata_and_bm25_fields() -> None:
    record = build_chroma_record(build_test_chunk())

    assert record.id == "chunk:50000000-0000-4000-8000-000000000001"
    assert "metoprolol 25 mg BID" in record.document
    assert "chest pressure" in record.document
    assert record.metadata["schema_version"] == HYBRID_INDEX_SCHEMA_VERSION
    assert record.metadata["patient_ref"] == "PATIENT_REF_0042"
    assert record.metadata["document_id"] == "40000000-0000-4000-8000-000000000001"
    assert record.metadata["parent_chunk_id"] == "50000000-0000-4000-8000-000000000000"
    assert record.metadata["section"] == "Medications"
    assert record.metadata["icd_codes"] == "I10"
    assert record.metadata["visit_date"] == "2025-02-14"
    assert record.metadata["bm25_medications"] == "metoprolol 25 mg BID"
    assert record.metadata["bm25_symptoms"] == "chest pressure"
    assert all(isinstance(value, str | int | float | bool) for value in record.metadata.values())


def test_build_chroma_upsert_payload_includes_embeddings_when_supplied() -> None:
    chunk = build_test_chunk()

    payload = build_chroma_upsert_payload([chunk], embeddings=[[0.1, 0.2, 0.3]])

    assert payload["ids"] == ["chunk:50000000-0000-4000-8000-000000000001"]
    assert payload["embeddings"] == [[0.1, 0.2, 0.3]]
    assert payload["metadatas"][0]["chunk_type"] == "child"


def test_ensure_chroma_collection_uses_mapping_metadata_without_real_client() -> None:
    fake_client = FakeChromaClient()

    collection = ensure_chroma_collection(
        client=fake_client,
        collection_name="unit_test_chunks",
    )

    assert collection.name == "unit_test_chunks"
    assert fake_client.created_collection_name == "unit_test_chunks"
    assert fake_client.created_collection_metadata is not None
    assert fake_client.created_collection_metadata["schema_version"] == HYBRID_INDEX_SCHEMA_VERSION
