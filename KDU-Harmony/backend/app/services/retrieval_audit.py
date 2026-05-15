from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.audit_event import AuditEvent
from app.models.enums import AuditAction
from app.models.user import User
from app.services.phi_store import role_name_values

RETRIEVAL_AUDIT_VERSION = "retrieval_audit_v1"


def record_rendered_retrieval_audit(
    db: Session,
    *,
    user: User,
    rendered_result: Any,
    ip_address: str | None = None,
    user_agent: str | None = None,
    commit: bool = True,
) -> list[AuditEvent]:
    recorded_at = datetime.now(UTC)
    role_snapshot = role_name_values(user)
    document_ids = result_document_ids(rendered_result)
    chunk_ids = result_chunk_ids(rendered_result)
    decision = access_decision(rendered_result)
    masking_mode = rendered_result.rendering_policy.render_mode
    masking_modes = hit_masking_modes(rendered_result)
    patient_assignment_modes = hit_patient_assignment_modes(rendered_result)
    filters = retrieval_filters(rendered_result)

    events = [
        AuditEvent(
            actor=user,
            action=AuditAction.QUERY_RUN,
            query_text=rendered_result.query,
            resource_type="retrieval_query",
            result_document_ids=document_ids,
            decision=decision,
            role_snapshot=role_snapshot,
            ip_address=ip_address,
            user_agent=user_agent,
            event_metadata={
                "audit_version": RETRIEVAL_AUDIT_VERSION,
                "timestamp": recorded_at.isoformat(),
                "user_id": str(user.id),
                "roles": role_snapshot,
                "query": rendered_result.query,
                "filters": filters,
                "document_ids": document_ids,
                "chunk_ids": chunk_ids,
                "masking_mode": masking_mode,
                "masking_modes": masking_modes,
                "patient_assignment_modes": patient_assignment_modes,
                "phi_visibility": rendered_result.rendering_policy.phi_visibility,
                "access_decision": decision,
                "result_count": len(rendered_result.hits),
            },
        )
    ]
    events.extend(
        document_access_events(
            user=user,
            rendered_result=rendered_result,
            role_snapshot=role_snapshot,
            recorded_at=recorded_at,
            decision=decision,
            masking_mode=masking_mode,
            masking_modes=masking_modes,
            patient_assignment_modes=patient_assignment_modes,
            filters=filters,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    )
    db.add_all(events)
    if commit:
        db.commit()
    return events


def document_access_events(
    *,
    user: User,
    rendered_result: Any,
    role_snapshot: list[str],
    recorded_at: datetime,
    decision: str,
    masking_mode: str,
    masking_modes: list[str],
    patient_assignment_modes: list[str],
    filters: dict[str, Any],
    ip_address: str | None,
    user_agent: str | None,
) -> list[AuditEvent]:
    chunks_by_document = chunk_ids_by_document(rendered_result)
    events: list[AuditEvent] = []
    for document_id, chunk_ids in chunks_by_document.items():
        events.append(
            AuditEvent(
                actor=user,
                action=AuditAction.DOCUMENT_READ,
                query_text=rendered_result.query,
                resource_type="document",
                resource_id=document_id,
                result_document_ids=[str(document_id)],
                decision=decision,
                role_snapshot=role_snapshot,
                ip_address=ip_address,
                user_agent=user_agent,
                event_metadata={
                    "audit_version": RETRIEVAL_AUDIT_VERSION,
                    "timestamp": recorded_at.isoformat(),
                    "user_id": str(user.id),
                    "roles": role_snapshot,
                    "query": rendered_result.query,
                    "filters": filters,
                    "document_id": str(document_id),
                    "chunk_ids": chunk_ids,
                    "masking_mode": masking_mode,
                    "masking_modes": masking_modes,
                    "patient_assignment_modes": patient_assignment_modes,
                    "phi_visibility": rendered_result.rendering_policy.phi_visibility,
                    "access_decision": decision,
                },
            )
        )
    return events


def chunk_ids_by_document(rendered_result: Any) -> dict[uuid.UUID, list[str]]:
    grouped: dict[uuid.UUID, list[str]] = defaultdict(list)
    for hit in rendered_result.hits:
        document_id = hit.citation.document_id
        chunk_id = str(hit.matched_chunk.chunk_id)
        if chunk_id not in grouped[document_id]:
            grouped[document_id].append(chunk_id)
    return dict(grouped)


def result_document_ids(rendered_result: Any) -> list[str]:
    return [str(document_id) for document_id in chunk_ids_by_document(rendered_result)]


def result_chunk_ids(rendered_result: Any) -> list[str]:
    chunk_ids: list[str] = []
    for hit in rendered_result.hits:
        chunk_id = str(hit.matched_chunk.chunk_id)
        if chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    return chunk_ids


def hit_masking_modes(rendered_result: Any) -> list[str]:
    values = {
        hit.retrieval.get("rendering", {}).get("render_mode")
        for hit in rendered_result.hits
        if isinstance(hit.retrieval, dict)
    }
    modes = sorted(value for value in values if isinstance(value, str) and value)
    return modes or [rendered_result.rendering_policy.render_mode]


def hit_patient_assignment_modes(rendered_result: Any) -> list[str]:
    values = {
        hit.retrieval.get("rendering", {}).get("patient_assignment")
        for hit in rendered_result.hits
        if isinstance(hit.retrieval, dict)
    }
    return sorted(value for value in values if isinstance(value, str) and value)


def access_decision(rendered_result: Any) -> str:
    if rendered_result.authorization.denied:
        return "deny"
    return "allow"


def retrieval_filters(rendered_result: Any) -> dict[str, Any]:
    authorization = rendered_result.authorization
    return {
        "chroma_where": authorization.chroma_where,
        "authorization_filters": authorization.authorization_filters,
        "query_metadata_filters": authorization.query_metadata_filters,
        "allowed_sensitivity_levels": authorization.allowed_sensitivity_levels,
        "unmet_policy_requirements": authorization.unmet_policy_requirements,
        "deny_reason": authorization.deny_reason,
    }
