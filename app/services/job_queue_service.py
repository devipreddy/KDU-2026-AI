from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.config import Settings
from app.db.session import SessionLocal
from app.models.database import ProcessingJobRecord
from app.services.processing_service import ProcessingService


class JobQueueService:
    def __init__(self, settings: Settings, processing_service: ProcessingService) -> None:
        self.settings = settings
        self.processing_service = processing_service
        self.executor = ThreadPoolExecutor(
            max_workers=settings.processing_max_workers,
            thread_name_prefix="content-processor",
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self._started = False
        self._lock = threading.Lock()

    def startup(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._resume_unfinished_jobs()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def submit_upload(
        self,
        *,
        file_name: str,
        content: bytes,
        content_type: str | None,
        force_reprocess: bool = False,
    ) -> tuple[ProcessingJobRecord, bool]:
        with SessionLocal() as db:
            file_record, cached = self.processing_service.prepare_upload(
                db=db,
                file_name=file_name,
                content=content,
                content_type=content_type,
                force_reprocess=force_reprocess,
            )
            if cached:
                job = ProcessingJobRecord(
                    id=uuid.uuid4().hex,
                    file_id=file_record.id,
                    file_name=file_record.file_name,
                    sha256=file_record.sha256,
                    status="completed",
                    progress_message="Reused cached processing results",
                    force_reprocess=force_reprocess,
                    metadata_json={"cached": True, "file_type": file_record.file_type},
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
                db.add(job)
                db.commit()
                db.refresh(job)
                return job, True

            job = ProcessingJobRecord(
                id=uuid.uuid4().hex,
                file_id=file_record.id,
                file_name=file_record.file_name,
                sha256=file_record.sha256,
                status="queued",
                progress_message="Queued for background processing",
                force_reprocess=force_reprocess,
                metadata_json={"cached": False, "file_type": file_record.file_type},
            )
            db.add(job)
            db.commit()
            db.refresh(job)

        self.executor.submit(self._run_job, job.id)
        return job, False

    def get_job(self, job_id: str) -> ProcessingJobRecord:
        with SessionLocal() as db:
            job = db.scalar(select(ProcessingJobRecord).where(ProcessingJobRecord.id == job_id))
            if job is None:
                raise ValueError("Processing job not found.")
            return job

    def _run_job(self, job_id: str) -> None:
        self._update_job(
            job_id,
            status="running",
            progress_message="Background worker started",
            started_at=datetime.now(timezone.utc),
            error_message=None,
        )
        try:
            job = self.get_job(job_id)
            if not job.file_id:
                raise RuntimeError("Processing job is missing its file reference.")

            def report_progress(message: str) -> None:
                self._update_job(job_id, status="running", progress_message=message)

            with SessionLocal() as db:
                self.processing_service.run_processing(
                    db=db,
                    file_id=job.file_id,
                    progress_callback=report_progress,
                )

            self._update_job(
                job_id,
                status="completed",
                progress_message="Processing complete",
                completed_at=datetime.now(timezone.utc),
            )
        except Exception as exc:  # pragma: no cover - background execution branch
            self.logger.exception("Job %s failed: %s", job_id, exc)
            self._update_job(
                job_id,
                status="failed",
                progress_message="Processing failed",
                error_message=str(exc),
                completed_at=datetime.now(timezone.utc),
            )

    def _update_job(self, job_id: str, **updates) -> None:
        with SessionLocal() as db:
            job = db.scalar(select(ProcessingJobRecord).where(ProcessingJobRecord.id == job_id))
            if job is None:
                return
            for key, value in updates.items():
                setattr(job, key, value)
            db.add(job)
            db.commit()

    def _resume_unfinished_jobs(self) -> None:
        with SessionLocal() as db:
            jobs = db.scalars(
                select(ProcessingJobRecord).where(ProcessingJobRecord.status.in_(("queued", "running")))
            ).all()
            for job in jobs:
                job.status = "queued"
                job.progress_message = "Resumed after application restart"
                db.add(job)
            db.commit()

        for job in jobs:
            self.executor.submit(self._run_job, job.id)
