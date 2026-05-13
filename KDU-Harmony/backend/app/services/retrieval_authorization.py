from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.session import SessionLocal
from app.models.access_policy import AccessPolicy
from app.models.enums import AccessPolicyEffect, SensitivityLevel
from app.models.user import User
from app.services.query_understanding import (
    DiagnosisConcept,
    QueryUnderstandingResult,
    understand_query,
)

DENY_ALL_WHERE = {"sensitivity_level": {"$eq": "__unauthorized__"}}
FILTER_VERSION = "authorization_aware_metadata_filter_v1"


@dataclass(frozen=True)
class DocumentAccessPolicy:
    policy_id: uuid.UUID
    role_name: str
    policy_name: str
    effect: str
    priority: int
    sensitivity_levels: list[str]
    phi_visibility: str | None
    conditions: dict[str, Any]


@dataclass(frozen=True)
class AuthorizedMetadataFilter:
    user_id: uuid.UUID
    role_names: list[str]
    policy_names: list[str]
    allowed_sensitivity_levels: list[str]
    phi_visibility: str | None
    chroma_where: dict[str, Any]
    query_metadata_filters: dict[str, Any]
    authorization_filters: dict[str, Any]
    denied: bool
    deny_reason: str | None
    unmet_policy_requirements: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "filter_version": FILTER_VERSION,
            "user_id": str(self.user_id),
            "role_names": self.role_names,
            "policy_names": self.policy_names,
            "allowed_sensitivity_levels": self.allowed_sensitivity_levels,
            "phi_visibility": self.phi_visibility,
            "chroma_where": self.chroma_where,
            "query_metadata_filters": self.query_metadata_filters,
            "authorization_filters": self.authorization_filters,
            "denied": self.denied,
            "deny_reason": self.deny_reason,
            "unmet_policy_requirements": self.unmet_policy_requirements,
        }


def build_authorized_metadata_filter(
    db: Session,
    *,
    user: User,
    query: QueryUnderstandingResult | str | None = None,
    query_metadata_filters: dict[str, Any] | None = None,
    authorized_patient_refs: list[str] | None = None,
) -> AuthorizedMetadataFilter:
    parsed_query = understand_query(query) if isinstance(query, str) else query
    query_filters = query_metadata_filters or (
        parsed_query.metadata_filters if parsed_query is not None else {}
    )
    policies = document_access_policies_for_user(db, user)
    role_names = role_name_values(user)

    if not policies:
        return deny_all_filter(
            user=user,
            role_names=role_names,
            query_filters=query_filters,
            deny_reason="no_active_document_access_policy",
        )

    allowed_sensitivity_levels = allowed_sensitivity_levels_from_policies(policies)
    if not allowed_sensitivity_levels:
        return deny_all_filter(
            user=user,
            role_names=role_names,
            query_filters=query_filters,
            policy_names=[policy.policy_name for policy in policies],
            deny_reason="no_allowed_sensitivity_levels",
        )

    unmet_requirements = unmet_policy_requirements(policies, authorized_patient_refs)
    authorization_filters = build_authorization_filter_payload(
        policies=policies,
        allowed_sensitivity_levels=allowed_sensitivity_levels,
        authorized_patient_refs=authorized_patient_refs,
    )
    query_predicates = metadata_filter_predicates(
        query_filters,
        diagnosis_concepts=parsed_query.diagnosis_concepts if parsed_query is not None else [],
    )
    auth_predicates = metadata_filter_predicates(authorization_filters)
    chroma_where = combine_chroma_predicates([*auth_predicates, *query_predicates])
    denied = chroma_where == DENY_ALL_WHERE

    return AuthorizedMetadataFilter(
        user_id=user.id,
        role_names=role_names,
        policy_names=[policy.policy_name for policy in policies],
        allowed_sensitivity_levels=allowed_sensitivity_levels,
        phi_visibility=highest_phi_visibility(policies),
        chroma_where=chroma_where,
        query_metadata_filters=query_filters,
        authorization_filters=authorization_filters,
        denied=denied,
        deny_reason="query_outside_authorized_scope" if denied else None,
        unmet_policy_requirements=unmet_requirements,
    )


def document_access_policies_for_user(db: Session, user: User) -> list[DocumentAccessPolicy]:
    role_ids = [role.id for role in user.roles]
    if not role_ids:
        return []

    policies = db.scalars(
        select(AccessPolicy)
        .options(selectinload(AccessPolicy.role))
        .where(AccessPolicy.role_id.in_(role_ids))
        .where(AccessPolicy.resource_type == "document")
        .where(AccessPolicy.is_active.is_(True))
        .order_by(AccessPolicy.priority, AccessPolicy.created_at)
    ).all()

    return [
        DocumentAccessPolicy(
            policy_id=policy.id,
            role_name=policy.role.name.value,
            policy_name=policy.name,
            effect=policy.effect.value,
            priority=policy.priority,
            sensitivity_levels=normalize_string_list(policy.conditions.get("sensitivity_levels")),
            phi_visibility=string_or_none(policy.conditions.get("phi_visibility")),
            conditions=policy.conditions,
        )
        for policy in policies
    ]


def allowed_sensitivity_levels_from_policies(policies: list[DocumentAccessPolicy]) -> list[str]:
    allowed: set[str] = set()
    denied: set[str] = set()

    for policy in policies:
        if policy.effect == AccessPolicyEffect.ALLOW.value:
            allowed.update(policy.sensitivity_levels)
        if policy.effect == AccessPolicyEffect.DENY.value:
            denied.update(policy.sensitivity_levels)
            denied.update(normalize_string_list(policy.conditions.get("deny_sensitivity_levels")))

    valid_order = [level.value for level in SensitivityLevel]
    return [level for level in valid_order if level in allowed and level not in denied]


def build_authorization_filter_payload(
    *,
    policies: list[DocumentAccessPolicy],
    allowed_sensitivity_levels: list[str],
    authorized_patient_refs: list[str] | None,
) -> dict[str, Any]:
    filters: dict[str, Any] = {
        "sensitivity_level": allowed_sensitivity_levels,
    }

    policy_patient_refs = intersect_policy_values(policies, "patient_refs")
    if authorized_patient_refs is not None:
        normalized_patient_refs = normalize_string_list(authorized_patient_refs)
        policy_patient_refs = (
            sorted(set(policy_patient_refs).intersection(normalized_patient_refs))
            if policy_patient_refs
            else normalized_patient_refs
        )
    if policy_patient_refs:
        filters["patient_ref"] = policy_patient_refs

    for metadata_field, condition_key in (
        ("hospital", "hospitals"),
        ("department", "departments"),
        ("document_type", "document_types"),
    ):
        policy_values = intersect_policy_values(policies, condition_key)
        if policy_values:
            filters[metadata_field] = policy_values

    return filters


def metadata_filter_predicates(
    filters: dict[str, Any],
    *,
    diagnosis_concepts: list[DiagnosisConcept] | None = None,
) -> list[dict[str, Any]]:
    predicates: list[dict[str, Any]] = []
    normalized_filters = dict(filters)
    if diagnosis_concepts:
        diagnosis_values = normalized_filters.get("diagnosis")
        expanded_diagnoses = normalize_string_list(diagnosis_values)
        has_exact_diagnosis = bool(expanded_diagnoses)
        for concept in diagnosis_concepts:
            if concept.concept_type == "diagnosis":
                expanded_diagnoses.extend(concept.diagnoses)
            elif not has_exact_diagnosis:
                expanded_diagnoses.extend(concept.diagnoses)
        if expanded_diagnoses:
            normalized_filters["diagnosis"] = sorted(set(expanded_diagnoses))

    for field, raw_value in normalized_filters.items():
        if field == "visit_date":
            predicates.extend(temporal_filter_predicates(raw_value))
            continue
        if field == "diagnosis_category":
            continue

        values = normalize_string_list(raw_value)
        if not values:
            continue
        predicates.append(field_predicate(field, values))
    return predicates


def temporal_filter_predicates(raw_filters: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_filters, list):
        return []

    predicates: list[dict[str, Any]] = []
    for raw_filter in raw_filters:
        if not isinstance(raw_filter, dict):
            continue
        operator = str(raw_filter.get("operator") or "").lower()
        start_date = raw_filter.get("start_date")
        end_date = raw_filter.get("end_date")
        if operator == "eq" and start_date:
            predicates.append({"visit_date": {"$eq": start_date}})
            continue
        if operator == "gte" and start_date:
            predicates.append({"visit_date": {"$gte": start_date}})
            continue
        if operator == "lte" and end_date:
            predicates.append({"visit_date": {"$lte": end_date}})
            continue
        if start_date:
            predicates.append({"visit_date": {"$gte": start_date}})
        if end_date:
            predicates.append({"visit_date": {"$lte": end_date}})
    return predicates


def field_predicate(field: str, values: list[str]) -> dict[str, Any]:
    if len(values) == 1:
        return {field: {"$eq": values[0]}}
    return {field: {"$in": values}}


def combine_chroma_predicates(predicates: list[dict[str, Any]]) -> dict[str, Any]:
    if any(predicate == DENY_ALL_WHERE for predicate in predicates):
        return DENY_ALL_WHERE
    if not predicates:
        return {}

    collapsed = collapse_equal_field_predicates(predicates)
    if collapsed is None:
        return DENY_ALL_WHERE
    if len(collapsed) == 1:
        return collapsed[0]
    return {"$and": collapsed}


def collapse_equal_field_predicates(
    predicates: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    exact_values_by_field: dict[str, set[str]] = {}
    passthrough: list[dict[str, Any]] = []

    for predicate in predicates:
        if len(predicate) != 1:
            passthrough.append(predicate)
            continue
        field, condition = next(iter(predicate.items()))
        if not isinstance(condition, dict) or set(condition) - {"$eq", "$in"}:
            passthrough.append(predicate)
            continue

        values = set(normalize_string_list(condition.get("$in") or condition.get("$eq")))
        if not values:
            continue
        if field in exact_values_by_field:
            exact_values_by_field[field] = exact_values_by_field[field].intersection(values)
        else:
            exact_values_by_field[field] = values

        if not exact_values_by_field[field]:
            return None

    collapsed = [
        field_predicate(field, sorted(values)) for field, values in exact_values_by_field.items()
    ]
    return [*collapsed, *passthrough]


def highest_phi_visibility(policies: list[DocumentAccessPolicy]) -> str | None:
    visibility_rank = {
        "de_identified": 0,
        "metadata_only": 1,
        "limited": 2,
        "operational": 3,
        "full": 4,
    }
    values = [
        policy.phi_visibility
        for policy in policies
        if policy.effect == AccessPolicyEffect.ALLOW.value and policy.phi_visibility
    ]
    if not values:
        return None
    return max(values, key=lambda value: visibility_rank.get(value, -1))


def unmet_policy_requirements(
    policies: list[DocumentAccessPolicy],
    authorized_patient_refs: list[str] | None,
) -> list[str]:
    requirements: set[str] = set()
    for policy in policies:
        if policy.effect != AccessPolicyEffect.ALLOW.value:
            continue
        if (
            policy.conditions.get("requires_treatment_relationship")
            and authorized_patient_refs is None
        ):
            requirements.add("requires_treatment_relationship")
        if (
            policy.conditions.get("requires_care_team_assignment")
            and authorized_patient_refs is None
        ):
            requirements.add("requires_care_team_assignment")
    return sorted(requirements)


def intersect_policy_values(policies: list[DocumentAccessPolicy], condition_key: str) -> list[str]:
    allow_value_sets: list[set[str]] = []
    deny_values: set[str] = set()

    for policy in policies:
        values = set(normalize_string_list(policy.conditions.get(condition_key)))
        if not values:
            continue
        if policy.effect == AccessPolicyEffect.ALLOW.value:
            allow_value_sets.append(values)
        if policy.effect == AccessPolicyEffect.DENY.value:
            deny_values.update(values)

    if not allow_value_sets:
        return []

    allowed_values = set.union(*allow_value_sets) - deny_values
    return sorted(allowed_values)


def deny_all_filter(
    *,
    user: User,
    role_names: list[str],
    query_filters: dict[str, Any],
    deny_reason: str,
    policy_names: list[str] | None = None,
) -> AuthorizedMetadataFilter:
    return AuthorizedMetadataFilter(
        user_id=user.id,
        role_names=role_names,
        policy_names=policy_names or [],
        allowed_sensitivity_levels=[],
        phi_visibility=None,
        chroma_where=DENY_ALL_WHERE,
        query_metadata_filters=query_filters,
        authorization_filters={},
        denied=True,
        deny_reason=deny_reason,
        unmet_policy_requirements=[],
    )


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set | frozenset):
        return sorted({str(item) for item in value if item is not None and str(item)})
    return [str(value)]


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def role_name_values(user: User) -> list[str]:
    return sorted(role.name.value for role in user.roles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build authorization-aware Chroma filters.")
    parser.add_argument("email")
    parser.add_argument("query", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        user = db.scalar(
            select(User).options(selectinload(User.roles)).where(User.email == args.email)
        )
        if user is None:
            raise SystemExit(f"User not found: {args.email}")
        result = build_authorized_metadata_filter(db, user=user, query=" ".join(args.query))
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
