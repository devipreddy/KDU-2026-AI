from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.models.database import ApiUsageRecord, ChunkRecord, FileRecord
from app.services.chunking import TokenChunker
from app.services.embedding_service import EmbeddingService
from app.services.enrichment_service import EnrichmentService
from app.services.presenters import usage_to_record_payload
from app.services.processors.audio_processor import AudioProcessor
from app.services.processors.image_processor import ImageProcessor
from app.services.processors.pdf_processor import PDFProcessor
from app.services.storage import StorageService
from app.services.text_normalizer import TextNormalizer
from app.services.types import ChunkPayload, ExtractionResult
from app.utils.files import compute_sha256, detect_file_type


class ProcessingService:
    def __init__(
        self,
        settings: Settings,
        pdf_processor: PDFProcessor,
        image_processor: ImageProcessor,
        audio_processor: AudioProcessor,
        normalizer: TextNormalizer,
        chunker: TokenChunker,
        enrichment_service: EnrichmentService,
        embedding_service: EmbeddingService,
        storage_service: StorageService,
    ) -> None:
        self.settings = settings
        self.pdf_processor = pdf_processor
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.normalizer = normalizer
        self.chunker = chunker
        self.enrichment_service = enrichment_service
        self.embedding_service = embedding_service
        self.storage_service = storage_service

    def prepare_upload(
        self,
        *,
        db: Session,
        file_name: str,
        content: bytes,
        content_type: str | None,
        force_reprocess: bool = False,
    ) -> tuple[FileRecord, bool]:
        if len(content) > self.settings.max_upload_size_bytes:
            raise ValueError(
                f"File exceeds max size of {self.settings.max_upload_size_mb} MB."
            )

        file_type, mime_type = detect_file_type(file_name, content_type)
        sha256 = compute_sha256(content)
        existing = db.scalar(
            select(FileRecord)
            .where(FileRecord.sha256 == sha256)
            .options(selectinload(FileRecord.chunks), selectinload(FileRecord.api_calls))
        )
        if existing and existing.status == "completed" and not force_reprocess:
            return existing, True

        stored_path = self.storage_service.save_upload(file_name=file_name, sha256=sha256, content=content)

        record = existing or FileRecord(
            id=uuid.uuid4().hex,
            file_name=file_name,
            stored_path=str(stored_path),
            mime_type=mime_type,
            file_type=file_type,
            size_bytes=len(content),
            sha256=sha256,
            status="queued",
        )
        record.file_name = file_name
        record.stored_path = str(stored_path)
        record.mime_type = mime_type
        record.file_type = file_type
        record.size_bytes = len(content)
        record.sha256 = sha256
        record.status = "queued"
        record.title = None
        record.description = None
        record.extracted_text = None
        record.cleaned_text = None
        record.summary = None
        record.key_points = []
        record.topic_tags = []
        record.extraction_metadata = {}
        record.chunk_count = 0
        record.processing_error = None
        record.processed_at = None

        if existing is None:
            db.add(record)
        else:
            db.add(record)

        db.commit()
        db.refresh(record)
        return record, False

    def run_processing(
        self,
        *,
        db: Session,
        file_id: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> FileRecord:
        record = self.get_file(db, file_id)
        record.status = "processing"
        record.processing_error = None
        db.add(record)
        db.commit()

        if progress_callback:
            progress_callback("Extracting content")

        self._clear_existing_artifacts(db, record.id)

        try:
            extraction = self._extract_content(file_type=record.file_type, file_path=Path(record.stored_path))
            if progress_callback:
                progress_callback("Normalizing transcript")
            normalized = self.normalizer.normalize(extraction.sections)
            if not normalized.cleaned_text:
                raise ValueError("No readable text was extracted from the file.")

            if progress_callback:
                progress_callback("Generating summary, key points, and tags")
            enrichment = self.enrichment_service.enrich(normalized.cleaned_text)
            if progress_callback:
                progress_callback("Building semantic search index")
            chunks = self.chunker.chunk_sections(
                sections=normalized.sections,
                file_id=record.id,
                max_tokens=self.settings.embedding_chunk_tokens,
                overlap_tokens=self.settings.embedding_chunk_overlap_tokens,
            )

            chunk_usage = self.embedding_service.index_chunks(file_name=record.file_name, chunks=chunks) if chunks else []

            record.title = enrichment.title
            record.description = extraction.description or self._build_description(extraction)
            record.extracted_text = normalized.extracted_text
            record.cleaned_text = normalized.cleaned_text
            record.summary = enrichment.summary
            record.key_points = enrichment.key_points
            record.topic_tags = enrichment.topic_tags
            record.chunk_count = len(chunks)
            record.status = "completed"
            record.processed_at = datetime.now(timezone.utc)
            record.extraction_metadata = {
                **extraction.metadata,
                **normalized.metadata,
                "sources": self._source_breakdown(normalized.sections),
            }

            for chunk in chunks:
                db.add(
                    ChunkRecord(
                        id=chunk.chunk_id,
                        file_id=record.id,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        token_count=chunk.token_count,
                        content=chunk.content,
                        metadata_json=chunk.metadata,
                    )
                )

            for usage in [*extraction.usage_entries, *enrichment.usage_entries, *chunk_usage]:
                db.add(ApiUsageRecord(**usage_to_record_payload(usage, file_id=record.id)))

            db.add(record)
            db.commit()
            db.refresh(record)
            if progress_callback:
                progress_callback("Processing complete")
            return self.get_file(db, record.id)
        except Exception as exc:
            self.embedding_service.vector_store.delete_file_chunks(record.id)
            db.execute(delete(ChunkRecord).where(ChunkRecord.file_id == record.id))
            db.execute(delete(ApiUsageRecord).where(ApiUsageRecord.file_id == record.id))
            record.status = "failed"
            record.processing_error = str(exc)
            db.add(record)
            db.commit()
            raise

    def process_upload(
        self,
        *,
        db: Session,
        file_name: str,
        content: bytes,
        content_type: str | None,
        force_reprocess: bool = False,
    ) -> tuple[FileRecord, bool]:
        record, cached = self.prepare_upload(
            db=db,
            file_name=file_name,
            content=content,
            content_type=content_type,
            force_reprocess=force_reprocess,
        )
        if cached:
            return record, True
        return self.run_processing(db=db, file_id=record.id), False

    def list_files(self, db: Session) -> list[FileRecord]:
        return db.scalars(
            select(FileRecord)
            .order_by(FileRecord.created_at.desc())
            .options(selectinload(FileRecord.chunks), selectinload(FileRecord.api_calls))
        ).all()

    def get_file(self, db: Session, file_id: str) -> FileRecord:
        record = db.scalar(
            select(FileRecord)
            .where(FileRecord.id == file_id)
            .options(selectinload(FileRecord.chunks), selectinload(FileRecord.api_calls))
        )
        if record is None:
            raise ValueError("File not found.")
        return record

    def _extract_content(self, *, file_type: str, file_path: Path) -> ExtractionResult:
        if file_type == "pdf":
            return self.pdf_processor.extract(file_path)
        if file_type == "image":
            return self.image_processor.extract(file_path)
        if file_type == "audio":
            return self.audio_processor.extract(file_path)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _build_description(self, extraction: ExtractionResult) -> str | None:
        descriptions = [section.description for section in extraction.sections if section.description]
        if not descriptions:
            return None
        return "\n\n".join(descriptions)

    def _source_breakdown(self, sections) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for section in sections:
            breakdown[section.source] = breakdown.get(section.source, 0) + 1
        return breakdown

    def _clear_existing_artifacts(self, db: Session, file_id: str) -> None:
        self.embedding_service.vector_store.delete_file_chunks(file_id)
        db.execute(delete(ChunkRecord).where(ChunkRecord.file_id == file_id))
        db.execute(delete(ApiUsageRecord).where(ApiUsageRecord.file_id == file_id))
        db.commit()
