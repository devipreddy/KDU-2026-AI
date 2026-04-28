from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_job_queue_service, get_processing_service
from app.api.security import require_api_key
from app.db.session import get_db
from app.schemas.file import (
    FileDetailResponse,
    FileProcessAcceptedResponse,
    FileSummaryResponse,
    ProcessingJobStatusResponse,
)
from app.services.job_queue_service import JobQueueService
from app.services.presenters import file_to_detail, file_to_summary, job_to_response
from app.services.processing_service import ProcessingService

router = APIRouter(prefix="/files", tags=["files"], dependencies=[Depends(require_api_key)])


@router.post("/process", response_model=FileProcessAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_file(
    file: UploadFile = File(...),
    force_reprocess: bool = Form(False),
    db: Session = Depends(get_db),
    queue: JobQueueService = Depends(get_job_queue_service),
    processing_service: ProcessingService = Depends(get_processing_service),
) -> FileProcessAcceptedResponse:
    try:
        content = await file.read()
        job, cached = queue.submit_upload(
            file_name=file.filename or "upload",
            content=content,
            content_type=file.content_type,
            force_reprocess=force_reprocess,
        )
        file_payload = None
        if job.file_id and job.status == "completed":
            file_payload = file_to_detail(processing_service.get_file(db, job.file_id))
        return FileProcessAcceptedResponse(
            cached=cached,
            job=job_to_response(job),
            file=file_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    finally:
        await file.close()


@router.get("/jobs/{job_id}", response_model=ProcessingJobStatusResponse)
def get_processing_job(
    job_id: str,
    db: Session = Depends(get_db),
    queue: JobQueueService = Depends(get_job_queue_service),
    processing_service: ProcessingService = Depends(get_processing_service),
) -> ProcessingJobStatusResponse:
    try:
        job = queue.get_job(job_id)
        file_payload = None
        if job.file_id and job.status == "completed":
            file_payload = file_to_detail(processing_service.get_file(db, job.file_id))
        return ProcessingJobStatusResponse(job=job_to_response(job), file=file_payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("", response_model=list[FileSummaryResponse])
def list_files(
    db: Session = Depends(get_db),
    service: ProcessingService = Depends(get_processing_service),
) -> list[FileSummaryResponse]:
    return [file_to_summary(record) for record in service.list_files(db)]


@router.get("/{file_id}", response_model=FileDetailResponse)
def get_file(
    file_id: str,
    db: Session = Depends(get_db),
    service: ProcessingService = Depends(get_processing_service),
) -> FileDetailResponse:
    try:
        return file_to_detail(service.get_file(db, file_id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
