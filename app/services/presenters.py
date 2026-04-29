from __future__ import annotations

from app.models.database import ApiUsageRecord, ChunkRecord, FileRecord, ProcessingJobRecord
from app.schemas.file import (
    ApiUsageResponse,
    ChunkResponse,
    FileDetailResponse,
    FileSummaryResponse,
    ProcessingJobResponse,
)


def file_to_summary(record: FileRecord) -> FileSummaryResponse:
    return FileSummaryResponse.model_validate(record)


def file_to_detail(record: FileRecord) -> FileDetailResponse:
    payload = FileDetailResponse.model_validate(record)
    payload.chunks = [ChunkResponse.model_validate(chunk) for chunk in sorted(record.chunks, key=lambda item: item.chunk_index)]
    payload.api_calls = [ApiUsageResponse.model_validate(call) for call in record.api_calls]
    return payload


def job_to_response(record: ProcessingJobRecord) -> ProcessingJobResponse:
    return ProcessingJobResponse.model_validate(record)


def usage_to_record_payload(entry, file_id: str | None) -> dict:
    return {
        "file_id": file_id,
        "operation": entry.operation,
        "provider": entry.provider,
        "model": entry.model,
        "input_tokens": entry.input_tokens,
        "output_tokens": entry.output_tokens,
        "total_tokens": entry.total_tokens,
        "estimated_cost_usd": entry.estimated_cost_usd,
        "response_ms": entry.response_ms,
        "metadata_json": entry.metadata,
    }
