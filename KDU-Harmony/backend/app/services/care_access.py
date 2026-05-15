from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.care_access import PatientCareAssignment, UserOrganizationScope
from app.models.enums import RoleName
from app.models.user import User

CARE_ACCESS_VERSION = "care_access_v1"
ASSIGNMENT_REQUIRED_ROLES = {RoleName.DOCTOR.value}


def organization_scopes_for_user(db: Session, user: User) -> list[str]:
    rows = db.scalars(
        select(UserOrganizationScope)
        .where(UserOrganizationScope.user_id == user.id)
        .where(UserOrganizationScope.is_active.is_(True))
        .order_by(UserOrganizationScope.is_primary.desc(), UserOrganizationScope.hospital)
    ).all()
    return sorted({row.hospital for row in rows if row.hospital})


def assigned_patient_refs_for_user(
    db: Session,
    user: User,
    *,
    hospitals: list[str] | None = None,
) -> set[str]:
    now = datetime.now(UTC)
    query = (
        select(PatientCareAssignment)
        .where(PatientCareAssignment.user_id == user.id)
        .where(PatientCareAssignment.is_active.is_(True))
        .where(
            (PatientCareAssignment.expires_at.is_(None)) | (PatientCareAssignment.expires_at >= now)
        )
    )
    if hospitals:
        query = query.where(
            (PatientCareAssignment.hospital.is_(None))
            | (PatientCareAssignment.hospital.in_(hospitals))
        )
    rows = db.scalars(query).all()
    return {row.patient_ref for row in rows if row.patient_ref}


def patient_is_assigned_to_user(
    db: Session,
    user: User,
    *,
    patient_ref: str | None,
    hospital: str | None = None,
) -> bool:
    if not patient_ref:
        return False
    hospitals = [hospital] if hospital else None
    return patient_ref in assigned_patient_refs_for_user(db, user, hospitals=hospitals)


def user_requires_patient_assignment_for_phi(user: User) -> bool:
    return bool(set(role_name_values(user)).intersection(ASSIGNMENT_REQUIRED_ROLES))


def role_name_values(user: User) -> list[str]:
    return sorted(role.name.value for role in user.roles)
