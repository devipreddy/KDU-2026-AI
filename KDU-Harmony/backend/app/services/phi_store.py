from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.access_policy import AccessPolicy
from app.models.audit_event import AuditEvent
from app.models.enums import AccessPolicyEffect, AuditAction, RoleName
from app.models.phi_mapping import PhiMapping
from app.models.user import User
from app.services.phi_tokenization import decrypt_phi_value

DIRECT_IDENTIFIER_TYPES = frozenset(
    {
        "PATIENT_NAME",
        "DOB",
        "MRN",
        "PHONE",
        "ADDRESS",
        "EMAIL",
    }
)

VISIBILITY_ALLOWED_ENTITY_TYPES = {
    "full": DIRECT_IDENTIFIER_TYPES,
    "operational": DIRECT_IDENTIFIER_TYPES,
    "limited": frozenset(),
    "metadata_only": frozenset(),
    "de_identified": frozenset(),
}

ROLE_VISIBILITY_FALLBACK = {
    RoleName.DOCTOR.value: "full",
    RoleName.NURSE.value: "limited",
    RoleName.ADMIN.value: "metadata_only",
    RoleName.RESEARCHER.value: "de_identified",
    RoleName.RECORDS_STAFF.value: "operational",
}


class PhiMappingNotFoundError(LookupError):
    """Raised when a PHI token does not resolve to a stored mapping."""


class PhiLookupDeniedError(PermissionError):
    """Raised when the current user is not allowed to decrypt a PHI mapping."""


@dataclass(frozen=True)
class PhiLookupResult:
    mapping_id: uuid.UUID
    token: str
    patient_ref: str
    entity_type: str
    value: str
    decrypted_at: datetime


def lookup_phi_value_for_user(
    db: Session,
    *,
    token: str,
    current_user: User,
    patient_ref: str | None = None,
) -> PhiLookupResult:
    normalized_token = token.strip()
    mapping = db.scalar(select(PhiMapping).where(PhiMapping.token == normalized_token))
    if mapping is None or (patient_ref is not None and mapping.patient_ref != patient_ref):
        record_phi_lookup_audit(
            db,
            actor=current_user,
            mapping=None,
            token=normalized_token,
            decision="not_found",
            reason="phi_mapping_not_found",
        )
        db.commit()
        raise PhiMappingNotFoundError("PHI token was not found")

    allowed_entity_types = allowed_phi_entity_types_for_user(db, current_user)
    if mapping.entity_type not in allowed_entity_types:
        record_phi_lookup_audit(
            db,
            actor=current_user,
            mapping=mapping,
            token=normalized_token,
            decision="deny",
            reason="phi_visibility_denied",
        )
        db.commit()
        raise PhiLookupDeniedError("PHI lookup is not permitted for this user")

    decrypted_at = datetime.now(UTC)
    value = decrypt_phi_value(mapping.encrypted_value, token=mapping.token)
    mapping.last_accessed_at = decrypted_at
    record_phi_lookup_audit(
        db,
        actor=current_user,
        mapping=mapping,
        token=normalized_token,
        decision="allow",
        reason="phi_visibility_allowed",
    )
    db.commit()
    return PhiLookupResult(
        mapping_id=mapping.id,
        token=mapping.token,
        patient_ref=mapping.patient_ref,
        entity_type=mapping.entity_type,
        value=value,
        decrypted_at=decrypted_at,
    )


def allowed_phi_entity_types_for_user(db: Session, user: User) -> frozenset[str]:
    visibility_values = phi_visibility_values_from_policies(db, user)
    if not visibility_values:
        visibility_values = [
            ROLE_VISIBILITY_FALLBACK[role_name]
            for role_name in role_name_values(user)
            if role_name in ROLE_VISIBILITY_FALLBACK
        ]

    allowed: set[str] = set()
    for visibility in visibility_values:
        allowed.update(VISIBILITY_ALLOWED_ENTITY_TYPES.get(visibility, frozenset()))
    return frozenset(allowed)


def phi_visibility_values_from_policies(db: Session, user: User) -> list[str]:
    role_ids = [role.id for role in user.roles]
    if not role_ids:
        return []

    policies = db.scalars(
        select(AccessPolicy)
        .where(AccessPolicy.role_id.in_(role_ids))
        .where(AccessPolicy.is_active.is_(True))
        .where(AccessPolicy.effect == AccessPolicyEffect.ALLOW)
    ).all()

    visibility_values: list[str] = []
    for policy in policies:
        visibility = policy.conditions.get("phi_visibility")
        if isinstance(visibility, str):
            visibility_values.append(visibility)
    return visibility_values


def record_phi_lookup_audit(
    db: Session,
    *,
    actor: User,
    mapping: PhiMapping | None,
    token: str,
    decision: str,
    reason: str,
) -> None:
    db.add(
        AuditEvent(
            actor=actor,
            action=AuditAction.PHI_DECRYPT,
            resource_type="phi_mapping",
            resource_id=mapping.id if mapping is not None else None,
            decision=decision,
            role_snapshot=role_name_values(actor),
            event_metadata={
                "token": token,
                "patient_ref": mapping.patient_ref if mapping is not None else None,
                "entity_type": mapping.entity_type if mapping is not None else None,
                "reason": reason,
            },
        )
    )


def role_name_values(user: User) -> list[str]:
    return sorted(role.name.value for role in user.roles)
