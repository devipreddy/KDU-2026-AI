from enum import StrEnum


def enum_values(enum_class: type[StrEnum]) -> list[str]:
    return [item.value for item in enum_class]


class RoleName(StrEnum):
    DOCTOR = "doctor"
    NURSE = "nurse"
    ADMIN = "admin"
    RESEARCHER = "researcher"
    RECORDS_STAFF = "records_staff"
    BILLING = "billing"
    SYSTEM = "system"


class DocumentType(StrEnum):
    TYPED_PDF = "typed_pdf"
    SCANNED_PDF = "scanned_pdf"
    CLINICAL_NOTE = "clinical_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    LAB_REPORT = "lab_report"
    PRESCRIPTION = "prescription"
    HANDWRITTEN_NOTE = "handwritten_note"
    OTHER = "other"


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    PROCESSED = "processed"
    REVIEW_REQUIRED = "review_required"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"


class SensitivityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    RESTRICTED = "restricted"


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class ChunkIndexingStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class AuditAction(StrEnum):
    LOGIN = "login"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_READ = "document_read"
    QUERY_RUN = "query_run"
    PHI_DECRYPT = "phi_decrypt"
    BREAK_GLASS = "break_glass"
    ACCESS_DENIED = "access_denied"


class AccessPolicyEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
