from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from auto_design.catalog.service import CatalogService
from auto_design.schemas.catalog import Product
from auto_design.schemas.intent import ColorRequest, StructuredIntent


CONFIDENCE_WEIGHTS: dict[str, float] = {
    "geometry": 0.4,
    "workflow": 0.3,
    "retrieval_color": 0.2,
    "topology": 0.1,
}

GEOMETRY_RULES = {
    "NKBA-CL-01",
    "NKBA-CL-02",
    "LAYOUT-01",
    "LAYOUT-02",
    "LAYOUT-03",
    "LAYOUT-04",
    "LAYOUT-05",
}
WORKFLOW_RULES = {"WORKFLOW-01", "WORKFLOW-02", "WORKFLOW-03"}
TOPOLOGY_RULES = {"LAYOUT-06"}


@dataclass(frozen=True)
class LayoutScore:
    variant_id: str
    score: float
    confidence: dict[str, float]
    weights: dict[str, float]
    violation_count: int
    hard_violation_count: int
    renderable: bool
    status: str

    def to_payload(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "score": self.score,
            "confidence": self.confidence,
            "weights": self.weights,
            "violation_count": self.violation_count,
            "hard_violation_count": self.hard_violation_count,
            "renderable": self.renderable,
            "status": self.status,
        }


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _variant_id(variant: Mapping[str, object]) -> str:
    return str(variant.get("id") or "variant")


def _violations(variant: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        violation for violation in _as_list(variant.get("violations"))
        if isinstance(violation, Mapping)
    ]


def _repair_history(variant: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        action for action in _as_list(variant.get("repair_history"))
        if isinstance(action, Mapping)
    ]


def _all_items(variant: Mapping[str, object]) -> list[Mapping[str, object]]:
    placement = _as_mapping(variant.get("placement"))
    items: list[Mapping[str, object]] = []
    for raw_run in _as_list(placement.get("runs")):
        run = _as_mapping(raw_run)
        items.extend(
            item for item in (_as_mapping(raw_item) for raw_item in _as_list(run.get("items")))
            if item
        )
    items.extend(
        item for item in (
            _as_mapping(raw_item) for raw_item in _as_list(placement.get("overhead_items"))
        )
        if item
    )
    if items:
        return items

    layout = _as_mapping(variant.get("layout"))
    return [
        item for item in (
            _as_mapping(raw_item) for raw_item in layout.values()
        )
        if item
    ]


def _severity(violation: Mapping[str, object]) -> str:
    return str(violation.get("severity") or "warning")


def _rule_id(violation: Mapping[str, object]) -> str:
    return str(violation.get("rule_id") or "")


def _penalty_for_severity(severity: str) -> float:
    if severity == "error":
        return 0.45
    if severity == "hard":
        return 0.3
    if severity == "soft":
        return 0.12
    return 0.07


def _repair_penalty(actions: list[Mapping[str, object]], rule_prefixes: tuple[str, ...]) -> float:
    matched = [
        action for action in actions
        if any(str(action.get("rule_id") or "").startswith(prefix) for prefix in rule_prefixes)
    ]
    return min(0.12, 0.025 * len(matched))


def _geometry_confidence(variant: Mapping[str, object]) -> float:
    placement = _as_mapping(variant.get("placement"))
    violations = _violations(variant)
    score = 1.0
    for violation in violations:
        rule_id = _rule_id(violation)
        if rule_id in GEOMETRY_RULES or _severity(violation) in {"hard", "error"}:
            score -= _penalty_for_severity(_severity(violation))

    if placement.get("is_continuous") is False:
        score -= 0.25
    if placement.get("base_coverage_valid") is False:
        score -= 0.25
    if not _as_mapping(variant.get("layout")):
        score -= 0.4

    score -= _repair_penalty(_repair_history(variant), ("NKBA", "LAYOUT"))
    return round(_clamp(score), 4)


def _workflow_confidence(variant: Mapping[str, object]) -> float:
    score = 1.0
    for violation in _violations(variant):
        if _rule_id(violation) in WORKFLOW_RULES:
            score -= _penalty_for_severity(_severity(violation))
    score -= _repair_penalty(_repair_history(variant), ("WORKFLOW",))
    return round(_clamp(score), 4)


def _product_for_item(
    catalog: CatalogService,
    item: Mapping[str, object],
) -> Product | None:
    product_id = item.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        return None
    if not catalog.has_sku(product_id):
        return None
    return catalog.get(product_id)


def _target_matches(product: Product, target: str) -> bool:
    if target == "base_cabinets":
        return product.category == "cabinet" and (
            product.type.startswith("base_") or "base_" in product.type
        )
    if target == "wall_cabinets":
        return product.category == "cabinet" and product.type.startswith("wall_")
    if target == "tall_cabinets":
        return product.category == "cabinet" and product.type.startswith("tall_")
    if target == "cabinets":
        return product.category == "cabinet"
    if target == "appliances":
        return product.category == "appliance"
    if target == "sinks":
        return product.type.startswith("sink_")
    if target == "fixtures":
        return product.category == "fixture" and not product.type.startswith("sink_")
    return True


def _color_request_score(
    request: ColorRequest,
    variant: Mapping[str, object],
    catalog: CatalogService,
) -> float:
    matched_skus = set(request.matched_skus)
    if not request.raw_text and not request.requested_hex and not request.resolved_hex:
        return 1.0
    if not matched_skus:
        return 0.45

    relevant_products = [
        product for product in (
            _product_for_item(catalog, item) for item in _all_items(variant)
        )
        if product is not None and _target_matches(product, request.target)
    ]
    if not relevant_products:
        return 0.75

    matched_count = sum(product.id in matched_skus for product in relevant_products)
    return matched_count / len(relevant_products)


def _retrieval_confidence(
    variant: Mapping[str, object],
    intent: StructuredIntent,
    catalog: CatalogService,
) -> float:
    if not intent.color_requests:
        return 1.0
    request_scores = [
        _color_request_score(request, variant, catalog)
        for request in intent.color_requests
    ]
    return round(_clamp(sum(request_scores) / len(request_scores)), 4)


def _topology_confidence(
    variant: Mapping[str, object],
    intent: StructuredIntent,
    feasibility: Mapping[str, object],
) -> float:
    expected_family = feasibility.get("selected_family") or intent.layout_family
    actual_family = variant.get("family")
    score = 1.0 if actual_family == expected_family else 0.45
    if variant.get("status") == "template_unavailable":
        score -= 0.5
    if feasibility.get("status") == "fallback":
        score -= 0.08
    for violation in _violations(variant):
        if _rule_id(violation) in TOPOLOGY_RULES:
            score -= _penalty_for_severity(_severity(violation))
    score -= _repair_penalty(_repair_history(variant), ("LAYOUT-06",))
    return round(_clamp(score), 4)


def _renderable(variant: Mapping[str, object]) -> bool:
    return bool(_as_mapping(variant.get("layout"))) and bool(_as_mapping(variant.get("placement")))


def _score_status(score: float, violations: list[Mapping[str, object]], renderable: bool) -> str:
    if not renderable:
        return "ranked_not_renderable"
    if violations:
        return "ranked_with_flags"
    if score >= 0.9:
        return "ranked_clean"
    return "ranked_after_tradeoffs"


def score_variant(
    variant: Mapping[str, object],
    *,
    intent: StructuredIntent,
    catalog: CatalogService,
    feasibility: Mapping[str, object],
) -> LayoutScore:
    confidence = {
        "geometry": _geometry_confidence(variant),
        "workflow": _workflow_confidence(variant),
        "retrieval_color": _retrieval_confidence(variant, intent, catalog),
        "topology": _topology_confidence(variant, intent, feasibility),
    }
    score = round(
        sum(confidence[key] * CONFIDENCE_WEIGHTS[key] for key in CONFIDENCE_WEIGHTS),
        4,
    )
    violations = _violations(variant)
    hard_count = sum(_severity(violation) in {"hard", "error"} for violation in violations)
    renderable = _renderable(variant)
    return LayoutScore(
        variant_id=_variant_id(variant),
        score=score,
        confidence=confidence,
        weights=dict(CONFIDENCE_WEIGHTS),
        violation_count=len(violations),
        hard_violation_count=hard_count,
        renderable=renderable,
        status=_score_status(score, violations, renderable),
    )


def rank_variants(
    variants: list[dict[str, object]],
    *,
    intent: StructuredIntent,
    catalog: CatalogService,
    feasibility: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    indexed_scores = [
        (
            index,
            score_variant(
                variant,
                intent=intent,
                catalog=catalog,
                feasibility=feasibility,
            ),
            variant,
        )
        for index, variant in enumerate(variants)
    ]
    indexed_scores.sort(
        key=lambda item: (
            -item[1].score,
            item[1].hard_violation_count,
            item[1].violation_count,
            item[0],
        )
    )

    ranked_variants: list[dict[str, object]] = []
    score_payloads: list[dict[str, object]] = []
    for rank, (_index, layout_score, variant) in enumerate(indexed_scores, start=1):
        scored_variant = copy.deepcopy(variant)
        scored_variant["score"] = layout_score.score
        scored_variant["confidence"] = layout_score.confidence
        scored_variant["confidence_weights"] = layout_score.weights
        scored_variant["rank"] = rank
        scored_variant["score_status"] = layout_score.status
        ranked_variants.append(scored_variant)
        payload = layout_score.to_payload()
        payload["rank"] = rank
        score_payloads.append(payload)

    return ranked_variants, score_payloads
