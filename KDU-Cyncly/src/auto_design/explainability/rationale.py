from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from auto_design.schemas.environment import Environment, Opening
from auto_design.schemas.intent import ColorRequest, StructuredIntent


SINK_COMPONENTS = {"sink", "single_sink", "double_sink"}
DISHWASHER_COMPONENTS = {"dishwasher"}
FRIDGE_CORNER_COMPONENTS = {"fridge", "tall_cabinet"}
DISHWASHER_SINK_MAX_GAP_MM = 600.0
FRIDGE_CORNER_TOLERANCE_MM = 600.0
WINDOW_SINK_TOLERANCE_MM = 300.0


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _rationale(
    rule_id: str,
    category: str,
    status: str,
    text: str,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_id": rule_id,
        "category": category,
        "status": status,
        "text": text,
    }
    if evidence:
        payload["evidence"] = dict(evidence)
    return payload


def _placement(variant: Mapping[str, object]) -> Mapping[str, object]:
    return _as_mapping(variant.get("placement"))


def _runs(variant: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        run
        for run in (_as_mapping(raw_run) for raw_run in _as_list(_placement(variant).get("runs")))
        if run
    ]


def _run_items(run: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        item
        for item in (_as_mapping(raw_item) for raw_item in _as_list(run.get("items")))
        if item
    ]


def _overhead_items(variant: Mapping[str, object]) -> list[Mapping[str, object]]:
    placement = _placement(variant)
    return [
        item
        for item in (
            _as_mapping(raw_item)
            for raw_item in _as_list(placement.get("overhead_items"))
        )
        if item
    ]


def _all_items(variant: Mapping[str, object]) -> list[Mapping[str, object]]:
    items: list[Mapping[str, object]] = []
    for run in _runs(variant):
        items.extend(_run_items(run))
    items.extend(_overhead_items(variant))
    return items


def _item_key(item: Mapping[str, object]) -> str:
    return str(item.get("key") or item.get("product_id") or "item")


def _component(item: Mapping[str, object]) -> str:
    return str(item.get("component") or "")


def _first_item(
    variant: Mapping[str, object],
    components: set[str],
) -> Mapping[str, object] | None:
    return next((item for item in _all_items(variant) if _component(item) in components), None)


def _run_for_item(
    variant: Mapping[str, object],
    target: Mapping[str, object],
) -> Mapping[str, object] | None:
    target_key = _item_key(target)
    for run in _runs(variant):
        if any(_item_key(item) == target_key for item in _run_items(run)):
            return run
    return None


def _item_center_offset(item: Mapping[str, object]) -> float | None:
    start = item.get("start_mm")
    end = item.get("end_mm")
    if isinstance(start, int | float) and isinstance(end, int | float):
        return (float(start) + float(end)) / 2.0
    return None


def _opening_center_offset(opening: Opening) -> float:
    return float(opening.offset_mm + (opening.width_mm / 2.0))


def _same_wall_gap(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> float | None:
    if first.get("wall") != second.get("wall"):
        return None
    first_start = first.get("start_mm")
    first_end = first.get("end_mm")
    second_start = second.get("start_mm")
    second_end = second.get("end_mm")
    values = (first_start, first_end, second_start, second_end)
    if not all(isinstance(value, int | float) for value in values):
        return None
    return max(
        float(second_start) - float(first_end),
        float(first_start) - float(second_end),
        0.0,
    )


def _xy_center(item: Mapping[str, object]) -> tuple[float, float]:
    position = _as_mapping(item.get("position_mm"))
    return (_as_float(position.get("x")), _as_float(position.get("y")))


def _xy_distance(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    return math.dist(_xy_center(first), _xy_center(second))


def _family_label(family: object) -> str:
    labels = {"I": "I-shaped", "L": "L-shaped", "U": "U-shaped"}
    return labels.get(str(family), str(family or "unknown"))


def _wall_list(variant: Mapping[str, object]) -> list[str]:
    topology = _as_mapping(variant.get("topology"))
    walls = [
        str(wall)
        for wall in _as_list(topology.get("walls"))
        if wall in {"north", "south", "east", "west"}
    ]
    if walls:
        return walls
    return [
        str(run.get("wall"))
        for run in _runs(variant)
        if run.get("wall") in {"north", "south", "east", "west"}
    ]


def _build_topology_rationale(
    variant: Mapping[str, object],
    intent: StructuredIntent,
    feasibility: Mapping[str, object],
) -> dict[str, object]:
    family = variant.get("family") or feasibility.get("selected_family") or intent.layout_family
    selected = feasibility.get("selected_family") or family
    requested = feasibility.get("requested_family") or intent.layout_family or selected
    walls = _wall_list(variant)
    status = "pass" if family == selected else "tradeoff"
    wall_text = ", ".join(walls) if walls else "no generated wall runs"
    text = (
        f"Selected {_family_label(family)} topology on {wall_text}; "
        f"requested family was {_family_label(requested)}."
    )
    if feasibility.get("status") == "fallback":
        status = "tradeoff"
        text += " Feasibility metadata records this as a fallback topology."
    return _rationale(
        "TOPOLOGY-01",
        "topology",
        status,
        text,
        {
            "template_id": str(variant.get("template_id") or ""),
            "requested_family": str(requested or ""),
            "selected_family": str(selected or ""),
            "walls": walls,
        },
    )


def _retrieved_product_count(
    retrieval_results: Mapping[str, object],
    matched_skus: list[str],
) -> int:
    sku_set = set(matched_skus)
    count = 0
    for raw_result in retrieval_results.values():
        result = _as_mapping(raw_result)
        products = [
            _as_mapping(product)
            for product in _as_list(result.get("products"))
        ]
        count += sum(str(product.get("id") or "") in sku_set for product in products)
    return count


def _color_request_rationale(
    request: ColorRequest,
    variant: Mapping[str, object],
    retrieval_results: Mapping[str, object],
) -> dict[str, object]:
    confidence = _as_float(_as_mapping(variant.get("confidence")).get("retrieval_color"), 1.0)
    matched_skus = list(request.matched_skus)
    sku_text = ", ".join(matched_skus) if matched_skus else "no catalog SKU"
    resolved = request.resolved_hex or request.requested_hex or "unresolved color"
    status = "pass" if matched_skus and confidence >= 0.95 else "tradeoff"
    text = (
        f"Prompt color '{request.raw_text}' resolved to {resolved} for "
        f"{request.target}; catalog candidate SKUs: {sku_text}."
    )
    if confidence < 0.95:
        text += f" Layout color confidence is {confidence:.2f}, so this variant is flagged."
    return _rationale(
        "COLOR-01",
        "retrieval",
        status,
        text,
        {
            "target": request.target,
            "requested_hex": request.requested_hex or "",
            "resolved_hex": request.resolved_hex or "",
            "matched_skus": matched_skus,
            "retrieved_candidate_count": _retrieved_product_count(
                retrieval_results,
                matched_skus,
            ),
            "confidence": confidence,
        },
    )


def _build_color_rationales(
    intent: StructuredIntent,
    variant: Mapping[str, object],
    retrieval_results: Mapping[str, object],
) -> list[dict[str, object]]:
    if not intent.color_requests:
        return [
            _rationale(
                "COLOR-DEFAULT",
                "retrieval",
                "pass",
                "No prompt color request was found; catalog default SKUs were preserved.",
            )
        ]
    return [
        _color_request_rationale(request, variant, retrieval_results)
        for request in intent.color_requests
    ]


def _build_sink_window_rationale(
    environment: Environment,
    variant: Mapping[str, object],
) -> dict[str, object]:
    windows = [opening for opening in environment.openings if opening.kind == "window"]
    sink = _first_item(variant, SINK_COMPONENTS)
    if not windows:
        return _rationale(
            "LAYOUT-01",
            "workflow",
            "not_applicable",
            "No window exists in the room; sink placement followed cleaning workflow.",
        )
    if sink is None:
        return _rationale(
            "LAYOUT-01",
            "workflow",
            "tradeoff",
            "A window exists, but no sink placement was available to align under it.",
        )

    sink_wall = str(sink.get("wall") or "")
    sink_offset = _item_center_offset(sink)
    same_wall_windows = [window for window in windows if window.wall == sink_wall]
    if sink_offset is None or not same_wall_windows:
        return _rationale(
            "LAYOUT-01",
            "workflow",
            "tradeoff",
            f"Sink {_item_key(sink)} is on {sink_wall}; no same-wall window center was usable.",
        )

    nearest = min(
        same_wall_windows,
        key=lambda opening: abs(_opening_center_offset(opening) - sink_offset),
    )
    delta = abs(_opening_center_offset(nearest) - sink_offset)
    status = "pass" if delta <= WINDOW_SINK_TOLERANCE_MM else "tradeoff"
    return _rationale(
        "LAYOUT-01",
        "workflow",
        status,
        (
            f"Sink {_item_key(sink)} is {delta:.0f} mm from window "
            f"{nearest.id} center on the {sink_wall} wall."
        ),
        {
            "sink_key": _item_key(sink),
            "window_id": nearest.id,
            "delta_mm": round(delta, 2),
            "tolerance_mm": WINDOW_SINK_TOLERANCE_MM,
        },
    )


def _build_dishwasher_rationale(variant: Mapping[str, object]) -> dict[str, object]:
    sink = _first_item(variant, SINK_COMPONENTS)
    dishwasher = _first_item(variant, DISHWASHER_COMPONENTS)
    if sink is None or dishwasher is None:
        return _rationale(
            "WORKFLOW-01",
            "workflow",
            "not_applicable",
            "Sink or dishwasher was not requested, so adjacency was not evaluated.",
        )

    gap = _same_wall_gap(sink, dishwasher)
    measured_as = "same-wall edge gap"
    if gap is None:
        gap = _xy_distance(sink, dishwasher)
        measured_as = "center distance"
    status = "pass" if gap <= DISHWASHER_SINK_MAX_GAP_MM else "tradeoff"
    return _rationale(
        "WORKFLOW-01",
        "workflow",
        status,
        (
            f"Dishwasher {_item_key(dishwasher)} is {gap:.0f} mm from sink "
            f"{_item_key(sink)} by {measured_as}."
        ),
        {
            "sink_key": _item_key(sink),
            "dishwasher_key": _item_key(dishwasher),
            "gap_mm": round(gap, 2),
            "limit_mm": DISHWASHER_SINK_MAX_GAP_MM,
            "measured_as": measured_as,
        },
    )


def _build_fridge_corner_rationale(variant: Mapping[str, object]) -> dict[str, object]:
    item = _first_item(variant, FRIDGE_CORNER_COMPONENTS)
    if item is None:
        return _rationale(
            "LAYOUT-06",
            "topology",
            "not_applicable",
            "No refrigerator or tall cabinet was placed in this variant.",
        )

    run = _run_for_item(variant, item)
    if run is None:
        return _rationale(
            "LAYOUT-06",
            "topology",
            "tradeoff",
            f"{_item_key(item)} was placed, but its wall run could not be resolved.",
        )

    distance = min(
        abs(_as_float(item.get("start_mm")) - _as_float(run.get("start_mm"))),
        abs(_as_float(run.get("end_mm")) - _as_float(item.get("end_mm"))),
    )
    status = "pass" if distance <= FRIDGE_CORNER_TOLERANCE_MM else "tradeoff"
    return _rationale(
        "LAYOUT-06",
        "topology",
        status,
        (
            f"{_item_key(item)} is {distance:.0f} mm from the nearest end of the "
            f"{run.get('wall')} run."
        ),
        {
            "item_key": _item_key(item),
            "component": _component(item),
            "wall": str(run.get("wall") or ""),
            "distance_mm": round(distance, 2),
            "tolerance_mm": FRIDGE_CORNER_TOLERANCE_MM,
        },
    )


def _build_repair_rationales(variant: Mapping[str, object]) -> list[dict[str, object]]:
    actions = [
        action
        for action in (
            _as_mapping(raw_action)
            for raw_action in _as_list(variant.get("repair_history"))
        )
        if action
    ]
    if not actions:
        return [
            _rationale(
                "REPAIR-00",
                "repair",
                "pass",
                "No local repair actions were required after validation.",
            )
        ]

    rationales: list[dict[str, object]] = []
    for action in actions:
        action_name = str(action.get("action") or "local_repair")
        rule_id = str(action.get("rule_id") or "REPAIR")
        item_keys = [
            str(item_key)
            for item_key in _as_list(action.get("item_keys"))
        ]
        rationales.append(
            _rationale(
                rule_id,
                "repair",
                "repaired",
                f"Repair '{action_name}' applied: {action.get('text') or ''}",
                {
                    "action": action_name,
                    "item_keys": item_keys,
                },
            )
        )
    return rationales


def _build_constraint_rationales(variant: Mapping[str, object]) -> list[dict[str, object]]:
    violations = [
        violation
        for violation in (
            _as_mapping(raw_violation)
            for raw_violation in _as_list(variant.get("violations"))
        )
        if violation
    ]
    rationales: list[dict[str, object]] = []
    if violations:
        for violation in violations:
            rationales.append(
                _rationale(
                    str(violation.get("rule_id") or "CONSTRAINT"),
                    "constraint",
                    "flagged",
                    str(violation.get("text") or "Constraint remains flagged."),
                    {
                        "severity": str(violation.get("severity") or "warning"),
                    },
                )
            )
    else:
        rationales.append(
            _rationale(
                "CONSTRAINT-00",
                "constraint",
                "pass",
                "No remaining validation violations after local repair and scoring.",
            )
        )

    confidence = _as_mapping(variant.get("confidence"))
    low_confidence = {
        str(key): _as_float(value)
        for key, value in confidence.items()
        if isinstance(value, int | float) and float(value) < 0.999
    }
    has_repairs = bool(_as_list(variant.get("repair_history")))
    if low_confidence or has_repairs:
        rationales.append(
            _rationale(
                "TRADEOFF-01",
                "constraint",
                "tradeoff",
                (
                    "Variant remains renderable with recorded tradeoffs across "
                    f"{', '.join(low_confidence) or 'repair history'}."
                ),
                {
                    "low_confidence": low_confidence,
                    "repair_count": len(_as_list(variant.get("repair_history"))),
                    "score_status": str(variant.get("score_status") or ""),
                },
            )
        )
    return rationales


def _build_scoring_rationale(variant: Mapping[str, object]) -> dict[str, object]:
    confidence = _as_mapping(variant.get("confidence"))
    confidence_text = ", ".join(
        f"{key}={_as_float(value):.2f}"
        for key, value in confidence.items()
        if isinstance(value, int | float)
    )
    score = _as_float(variant.get("score"))
    return _rationale(
        "SCORE-01",
        "scoring",
        str(variant.get("score_status") or "ranked"),
        f"Composite score is {score:.2f}; confidence breakdown: {confidence_text}.",
        {
            "score": score,
            "rank": _as_float(variant.get("rank")),
            "confidence": dict(confidence),
            "weights": dict(_as_mapping(variant.get("confidence_weights"))),
        },
    )


def build_variant_explainability_trace(
    variant: Mapping[str, object],
    *,
    intent: StructuredIntent,
    environment: Environment,
    feasibility: Mapping[str, object],
    retrieval_results: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build reviewer-facing reasoning for one ranked layout variant."""

    trace: list[dict[str, object]] = [
        _build_topology_rationale(variant, intent, feasibility),
        *_build_color_rationales(intent, variant, retrieval_results),
        _build_sink_window_rationale(environment, variant),
        _build_dishwasher_rationale(variant),
        _build_fridge_corner_rationale(variant),
        *_build_repair_rationales(variant),
        *_build_constraint_rationales(variant),
        _build_scoring_rationale(variant),
    ]
    return trace


def attach_rationales_to_variants(
    variants: list[dict[str, object]],
    *,
    intent: StructuredIntent,
    environment: Environment,
    feasibility: Mapping[str, object],
    retrieval_results: Mapping[str, object],
) -> list[dict[str, object]]:
    annotated_variants: list[dict[str, object]] = []
    for variant in variants:
        annotated = copy.deepcopy(variant)
        trace = build_variant_explainability_trace(
            annotated,
            intent=intent,
            environment=environment,
            feasibility=feasibility,
            retrieval_results=retrieval_results,
        )
        annotated["explainability_trace"] = trace
        annotated["rationale"] = [
            {
                "rule_id": str(entry["rule_id"]),
                "text": str(entry["text"]),
            }
            for entry in trace
        ]
        annotated_variants.append(annotated)
    return annotated_variants
