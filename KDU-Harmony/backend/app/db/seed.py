from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.access_policy import AccessPolicy
from app.models.care_access import PatientCareAssignment, UserOrganizationScope
from app.models.enums import AccessPolicyEffect, RoleName
from app.models.role import Role
from app.models.user import User

DEMO_PASSWORD = "ChangeMe123!"


@dataclass(frozen=True)
class SeedRole:
    id: UUID
    name: RoleName
    display_name: str
    description: str


@dataclass(frozen=True)
class SeedUser:
    id: UUID
    email: str
    display_name: str
    department: str
    role_name: RoleName


SEEDED_ROLES = (
    SeedRole(
        id=UUID("10000000-0000-4000-8000-000000000001"),
        name=RoleName.DOCTOR,
        display_name="Doctor",
        description="Treating clinician with full clinical context for authorized patients.",
    ),
    SeedRole(
        id=UUID("10000000-0000-4000-8000-000000000002"),
        name=RoleName.NURSE,
        display_name="Nurse",
        description="Care team user with limited clinical context.",
    ),
    SeedRole(
        id=UUID("10000000-0000-4000-8000-000000000003"),
        name=RoleName.ADMIN,
        display_name="Administrator",
        description="Platform administrator with metadata and system-management access.",
    ),
    SeedRole(
        id=UUID("10000000-0000-4000-8000-000000000004"),
        name=RoleName.RESEARCHER,
        display_name="Researcher",
        description="Research user restricted to de-identified output.",
    ),
    SeedRole(
        id=UUID("10000000-0000-4000-8000-000000000005"),
        name=RoleName.RECORDS_STAFF,
        display_name="Records Staff",
        description="Records operations user with document workflow access.",
    ),
)

SEEDED_USERS = (
    SeedUser(
        id=UUID("20000000-0000-4000-8000-000000000001"),
        email="doctor@example.com",
        display_name="Dr. Demo Doctor",
        department="Cardiology",
        role_name=RoleName.DOCTOR,
    ),
    SeedUser(
        id=UUID("20000000-0000-4000-8000-000000000002"),
        email="nurse@example.com",
        display_name="Nora Demo Nurse",
        department="Care Team",
        role_name=RoleName.NURSE,
    ),
    SeedUser(
        id=UUID("20000000-0000-4000-8000-000000000003"),
        email="admin@example.com",
        display_name="Ari Demo Admin",
        department="Platform Operations",
        role_name=RoleName.ADMIN,
    ),
    SeedUser(
        id=UUID("20000000-0000-4000-8000-000000000004"),
        email="researcher@example.com",
        display_name="Rhea Demo Researcher",
        department="Clinical Research",
        role_name=RoleName.RESEARCHER,
    ),
    SeedUser(
        id=UUID("20000000-0000-4000-8000-000000000005"),
        email="records@example.com",
        display_name="Remy Demo Records",
        department="Records Management",
        role_name=RoleName.RECORDS_STAFF,
    ),
)

POLICY_CONDITIONS = {
    RoleName.DOCTOR: {
        "phi_visibility": "full",
        "sensitivity_levels": ["low", "medium", "high", "restricted"],
        "requires_organization_scope": True,
        "requires_treatment_relationship": True,
    },
    RoleName.NURSE: {
        "phi_visibility": "limited",
        "sensitivity_levels": ["low", "medium", "high"],
        "requires_organization_scope": True,
        "requires_care_team_assignment": True,
    },
    RoleName.ADMIN: {
        "phi_visibility": "metadata_only",
        "sensitivity_levels": ["low", "medium", "high", "restricted"],
    },
    RoleName.RESEARCHER: {
        "phi_visibility": "de_identified",
        "sensitivity_levels": ["low", "medium"],
    },
    RoleName.RECORDS_STAFF: {
        "phi_visibility": "operational",
        "sensitivity_levels": ["low", "medium", "high"],
    },
}

SEEDED_ORGANIZATION_SCOPES = (
    {
        "id": UUID("40000000-0000-4000-8000-000000000001"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "hospital": "Harmony General Hospital",
        "is_primary": True,
    },
    {
        "id": UUID("40000000-0000-4000-8000-000000000002"),
        "user_id": UUID("20000000-0000-4000-8000-000000000002"),
        "hospital": "Harmony General Hospital",
        "is_primary": True,
    },
)

SEEDED_PATIENT_ASSIGNMENTS = (
    {
        "id": UUID("41000000-0000-4000-8000-000000000001"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_2301",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000002"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_0300",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000003"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_0100",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000004"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_0001",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000005"),
        "user_id": UUID("20000000-0000-4000-8000-000000000002"),
        "patient_ref": "PATIENT_REF_0042",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000006"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_0050",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000007"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_0201",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000008"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_0012",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": UUID("41000000-0000-4000-8000-000000000009"),
        "user_id": UUID("20000000-0000-4000-8000-000000000001"),
        "patient_ref": "PATIENT_REF_0229",
        "hospital": "Harmony General Hospital",
    },
)


def seeded_password_hash(role_name: RoleName) -> str:
    return hash_password(DEMO_PASSWORD, salt=f"local-dev-auth:{role_name.value}".encode())


def seed_auth_data(db: Session) -> None:
    roles_by_name: dict[RoleName, Role] = {}

    for seed_role in SEEDED_ROLES:
        role = db.scalar(select(Role).where(Role.name == seed_role.name))
        if role is None:
            role = Role(id=seed_role.id, name=seed_role.name)
        role.display_name = seed_role.display_name
        role.description = seed_role.description
        db.add(role)
        roles_by_name[seed_role.name] = role

    db.flush()

    for seed_user in SEEDED_USERS:
        user = db.scalar(select(User).where(User.email == seed_user.email))
        if user is None:
            user = User(id=seed_user.id, email=seed_user.email)
        user.display_name = seed_user.display_name
        user.department = seed_user.department
        user.password_hash = seeded_password_hash(seed_user.role_name)
        user.external_subject = f"local-dev:{seed_user.email}"
        user.is_active = True
        user.roles = [roles_by_name[seed_user.role_name]]
        db.add(user)

    for seed_role in SEEDED_ROLES:
        policy_name = f"{seed_role.name.value}_document_access"
        policy = db.scalar(
            select(AccessPolicy)
            .where(AccessPolicy.role_id == seed_role.id)
            .where(AccessPolicy.name == policy_name)
        )
        if policy is None:
            policy = AccessPolicy(
                id=UUID(f"30000000-0000-4000-8000-{seed_role.id.int % 1000000000000:012d}"),
                role_id=seed_role.id,
                name=policy_name,
            )
        policy.description = f"Default document access policy for {seed_role.display_name}."
        policy.effect = AccessPolicyEffect.ALLOW
        policy.resource_type = "document"
        policy.conditions = POLICY_CONDITIONS[seed_role.name]
        policy.priority = 100
        policy.is_active = True
        db.add(policy)

    for seed_scope in SEEDED_ORGANIZATION_SCOPES:
        scope = db.scalar(
            select(UserOrganizationScope).where(UserOrganizationScope.id == seed_scope["id"])
        )
        if scope is None:
            scope = UserOrganizationScope(id=seed_scope["id"])
        scope.user_id = seed_scope["user_id"]
        scope.hospital = seed_scope["hospital"]
        scope.is_primary = bool(seed_scope["is_primary"])
        scope.is_active = True
        db.add(scope)

    for seed_assignment in SEEDED_PATIENT_ASSIGNMENTS:
        assignment = db.scalar(
            select(PatientCareAssignment).where(PatientCareAssignment.id == seed_assignment["id"])
        )
        if assignment is None:
            assignment = PatientCareAssignment(id=seed_assignment["id"])
        assignment.user_id = seed_assignment["user_id"]
        assignment.patient_ref = seed_assignment["patient_ref"]
        assignment.hospital = seed_assignment["hospital"]
        assignment.is_active = True
        assignment.expires_at = None
        db.add(assignment)

    db.commit()


def main() -> None:
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        seed_auth_data(db)
    print(
        f"Seeded {len(SEEDED_ROLES)} roles and {len(SEEDED_USERS)} demo users. "
        f"Default password: {DEMO_PASSWORD}"
    )


if __name__ == "__main__":
    main()
