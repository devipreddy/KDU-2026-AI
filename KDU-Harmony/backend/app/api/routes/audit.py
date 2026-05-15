from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import require_roles
from app.db.session import get_db
from app.models.audit_event import AuditEvent
from app.models.enums import RoleName
from app.models.user import User
from app.schemas.audit import AuditEventResponse

router = APIRouter(prefix="/audit")

DbSession = Annotated[Session, Depends(get_db)]
AuditUser = Annotated[
    User,
    Depends(require_roles(RoleName.ADMIN.value, RoleName.RECORDS_STAFF.value)),
]


@router.get("/events", response_model=list[AuditEventResponse])
def list_audit_events(
    _current_user: AuditUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    q: str | None = None,
) -> list[AuditEventResponse]:
    events = db.scalars(
        select(AuditEvent)
        .options(selectinload(AuditEvent.actor))
        .order_by(AuditEvent.occurred_at.desc())
        .limit(limit)
    ).all()
    if q:
        needle = q.lower()
        events = [event for event in events if audit_event_matches(event, needle)]
    return [serialize_audit_event(event) for event in events]


def audit_event_matches(event: AuditEvent, needle: str) -> bool:
    actor = event.actor
    haystack = " ".join(
        [
            str(event.id),
            actor.email if actor else "",
            actor.display_name if actor else "",
            event.action.value,
            event.query_text or "",
            event.resource_type or "",
            str(event.resource_id or ""),
            " ".join(event.result_document_ids),
            event.decision or "",
            " ".join(event.role_snapshot),
            json.dumps(event.event_metadata, sort_keys=True),
        ]
    ).lower()
    return needle in haystack


def serialize_audit_event(event: AuditEvent) -> AuditEventResponse:
    actor = event.actor
    return AuditEventResponse(
        id=event.id,
        occurred_at=event.occurred_at,
        actor_email=actor.email if actor else None,
        actor_display_name=actor.display_name if actor else None,
        role_snapshot=event.role_snapshot,
        action=event.action.value,
        query_text=event.query_text,
        resource_type=event.resource_type,
        resource_id=event.resource_id,
        result_document_ids=event.result_document_ids,
        decision=event.decision,
        ip_address=event.ip_address,
        user_agent=event.user_agent,
        metadata=event.event_metadata,
    )
