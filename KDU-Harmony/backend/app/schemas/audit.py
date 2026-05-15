from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class AuditEventResponse(BaseModel):
    id: UUID
    occurred_at: datetime
    actor_email: str | None
    actor_display_name: str | None
    role_snapshot: list[str]
    action: str
    query_text: str | None
    resource_type: str | None
    resource_id: UUID | None
    result_document_ids: list[str]
    decision: str | None
    ip_address: str | None
    user_agent: str | None
    metadata: dict[str, Any]
