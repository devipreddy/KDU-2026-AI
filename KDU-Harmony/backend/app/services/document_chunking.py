from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.clinical_metadata import ClinicalMetadataResult, DocumentSection

CHUNKER_VERSION = "hierarchical_medical_chunker_v1"
DEFAULT_CHILD_TOKEN_TARGET = 80
DEFAULT_CHILD_TOKEN_OVERLAP = 16


@dataclass(frozen=True)
class PlannedChildChunk:
    content: str
    start_offset: int
    end_offset: int
    token_count: int


@dataclass(frozen=True)
class PlannedSectionChunk:
    section: str
    content: str
    start_offset: int
    end_offset: int
    token_count: int
    children: list[PlannedChildChunk]


@dataclass(frozen=True)
class ChunkingResult:
    parent_chunk_count: int
    child_chunk_count: int
    section_count: int
    chunk_ids: list[str]
    sections: list[str]

    @property
    def total_chunk_count(self) -> int:
        return self.parent_chunk_count + self.child_chunk_count

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunker": CHUNKER_VERSION,
            "parent_chunk_count": self.parent_chunk_count,
            "child_chunk_count": self.child_chunk_count,
            "total_chunk_count": self.total_chunk_count,
            "section_count": self.section_count,
            "sections": self.sections,
            "chunk_ids": self.chunk_ids,
            "child_token_target": DEFAULT_CHILD_TOKEN_TARGET,
            "child_token_overlap": DEFAULT_CHILD_TOKEN_OVERLAP,
        }


def chunk_document_text(
    db: Session,
    *,
    document: Document,
    text: str,
    clinical_metadata: ClinicalMetadataResult,
    ocr_confidence: float | None = None,
) -> ChunkingResult:
    delete_existing_chunks(db, document)
    planned_sections = plan_section_chunks(text, clinical_metadata.document_sections)

    chunk_index = 0
    chunk_ids: list[str] = []
    persisted_sections: list[str] = []
    parent_count = 0
    child_count = 0

    for planned_section in planned_sections:
        parent_chunk = build_document_chunk(
            document=document,
            chunk_index=chunk_index,
            section=planned_section.section,
            content=planned_section.content,
            start_offset=planned_section.start_offset,
            end_offset=planned_section.end_offset,
            token_count=planned_section.token_count,
            ocr_confidence=ocr_confidence,
            retrieval_metadata=base_retrieval_metadata(
                document=document,
                clinical_metadata=clinical_metadata,
                chunk_type="parent",
                section=planned_section.section,
                child_count=len(planned_section.children),
            ),
        )
        db.add(parent_chunk)
        chunk_ids.append(str(parent_chunk.id))
        persisted_sections.append(planned_section.section)
        parent_count += 1
        chunk_index += 1

        for child_order, planned_child in enumerate(planned_section.children):
            child_chunk = build_document_chunk(
                document=document,
                chunk_index=chunk_index,
                section=planned_section.section,
                content=planned_child.content,
                start_offset=planned_child.start_offset,
                end_offset=planned_child.end_offset,
                token_count=planned_child.token_count,
                ocr_confidence=ocr_confidence,
                parent_chunk=parent_chunk,
                retrieval_metadata=base_retrieval_metadata(
                    document=document,
                    clinical_metadata=clinical_metadata,
                    chunk_type="child",
                    section=planned_section.section,
                    child_order=child_order,
                    parent_chunk_id=str(parent_chunk.id),
                ),
            )
            db.add(child_chunk)
            chunk_ids.append(str(child_chunk.id))
            child_count += 1
            chunk_index += 1

    db.flush()
    return ChunkingResult(
        parent_chunk_count=parent_count,
        child_chunk_count=child_count,
        section_count=len(planned_sections),
        chunk_ids=chunk_ids,
        sections=ordered_unique(persisted_sections),
    )


def delete_existing_chunks(db: Session, document: Document) -> None:
    existing_chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(
            DocumentChunk.parent_chunk_id.desc().nullslast(), DocumentChunk.chunk_index.desc()
        )
    ).all()
    for chunk in existing_chunks:
        db.delete(chunk)
    if existing_chunks:
        db.flush()


def plan_section_chunks(
    text: str,
    document_sections: list[DocumentSection],
    *,
    child_token_target: int = DEFAULT_CHILD_TOKEN_TARGET,
    child_token_overlap: int = DEFAULT_CHILD_TOKEN_OVERLAP,
) -> list[PlannedSectionChunk]:
    sections = document_sections or [fallback_document_section(text)]
    planned_sections: list[PlannedSectionChunk] = []

    for section in sections:
        content = text[section.start_offset : section.end_offset].strip()
        if not content:
            continue
        start_offset = text.find(content, section.start_offset, section.end_offset)
        if start_offset < 0:
            start_offset = section.start_offset
        end_offset = start_offset + len(content)
        children = plan_child_chunks(
            text=text,
            section_content=content,
            section_start_offset=start_offset,
            child_token_target=child_token_target,
            child_token_overlap=child_token_overlap,
        )
        planned_sections.append(
            PlannedSectionChunk(
                section=section.section,
                content=content,
                start_offset=start_offset,
                end_offset=end_offset,
                token_count=count_tokens(content),
                children=children,
            )
        )

    if planned_sections:
        return planned_sections
    return [
        PlannedSectionChunk(
            section="Document",
            content=text.strip(),
            start_offset=0,
            end_offset=len(text.strip()),
            token_count=count_tokens(text),
            children=plan_child_chunks(
                text=text,
                section_content=text.strip(),
                section_start_offset=0,
                child_token_target=child_token_target,
                child_token_overlap=child_token_overlap,
            ),
        )
    ]


def plan_child_chunks(
    *,
    text: str,
    section_content: str,
    section_start_offset: int,
    child_token_target: int,
    child_token_overlap: int,
) -> list[PlannedChildChunk]:
    token_matches = list(re.finditer(r"\S+", section_content))
    if not token_matches:
        return []

    child_chunks: list[PlannedChildChunk] = []
    stride = max(1, child_token_target - child_token_overlap)
    start_token_index = 0

    while start_token_index < len(token_matches):
        end_token_index = min(start_token_index + child_token_target, len(token_matches))
        start_in_section = token_matches[start_token_index].start()
        end_in_section = token_matches[end_token_index - 1].end()
        absolute_start = section_start_offset + start_in_section
        absolute_end = section_start_offset + end_in_section
        content = text[absolute_start:absolute_end].strip()
        if content:
            content_start = text.find(content, absolute_start, absolute_end)
            if content_start < 0:
                content_start = absolute_start
            child_chunks.append(
                PlannedChildChunk(
                    content=content,
                    start_offset=content_start,
                    end_offset=content_start + len(content),
                    token_count=count_tokens(content),
                )
            )
        if end_token_index == len(token_matches):
            break
        start_token_index += stride

    return child_chunks


def build_document_chunk(
    *,
    document: Document,
    chunk_index: int,
    section: str,
    content: str,
    start_offset: int,
    end_offset: int,
    token_count: int,
    ocr_confidence: float | None,
    retrieval_metadata: dict[str, Any],
    parent_chunk: DocumentChunk | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        id=uuid.uuid4(),
        document=document,
        parent_chunk=parent_chunk,
        chunk_index=chunk_index,
        section=section,
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        embedding_collection=settings.chroma_collection,
        embedding_id=None,
        token_count=token_count,
        start_offset=start_offset,
        end_offset=end_offset,
        page_number=None,
        ocr_confidence=ocr_confidence,
        sensitivity_level=document.sensitivity_level,
        retrieval_metadata=retrieval_metadata,
    )


def base_retrieval_metadata(
    *,
    document: Document,
    clinical_metadata: ClinicalMetadataResult,
    chunk_type: str,
    section: str,
    **extra_metadata: Any,
) -> dict[str, Any]:
    metadata = {
        "chunker": CHUNKER_VERSION,
        "chunk_type": chunk_type,
        "section": section,
        "patient_ref": document.patient_ref,
        "visit_id": document.visit_id,
        "document_id": str(document.id),
        "external_id": document.external_id,
        "document_type": document.document_type.value,
        "hospital": document.hospital,
        "physician": document.physician,
        "diagnosis": document.diagnosis,
        "icd_codes": document.icd_codes,
        "sensitivity_level": document.sensitivity_level.value,
        "clinical_entities": {
            "diagnoses": clinical_metadata.diagnoses,
            "medications": clinical_metadata.medications,
            "symptoms": clinical_metadata.symptoms,
            "icd_codes": clinical_metadata.icd_codes,
            "dates": [asdict(date) for date in clinical_metadata.dates],
        },
    }
    metadata.update(extra_metadata)
    return metadata


def fallback_document_section(text: str) -> DocumentSection:
    return DocumentSection(
        section="Document",
        order=0,
        start_offset=0,
        end_offset=len(text),
        char_count=len(text),
    )


def count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(value)
    return unique_values
