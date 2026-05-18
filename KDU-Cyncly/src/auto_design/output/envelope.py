from __future__ import annotations

import json
import uuid
from collections.abc import Mapping

from auto_design.schemas.environment import Environment, Opening, Wall
from auto_design.schemas.input import DesignInput
from auto_design.schemas.output import LayoutResponse


DEFAULT_WALL_THICKNESS_MM = 100.0


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _points_extent(points: list[object], axis: str) -> tuple[float, float]:
    values = [
        _as_float(_as_mapping(point).get(axis))
        for point in points
    ]
    if not values:
        return (0.0, 0.0)
    return (min(values), max(values))


def _wall_length_mm(wall: Wall) -> float:
    if wall.dimensions.length_mm is not None:
        return float(wall.dimensions.length_mm)
    min_x, max_x = _points_extent(wall.model_dump(mode="json")["points"], "x")
    min_y, max_y = _points_extent(wall.model_dump(mode="json")["points"], "y")
    return max(max_x - min_x, max_y - min_y)


def _wall_layout_item(wall: Wall) -> dict[str, object]:
    wall_payload = wall.model_dump(mode="json")
    points = wall_payload["points"]
    min_x, max_x = _points_extent(points, "x")
    min_y, max_y = _points_extent(points, "y")
    height = float(wall.dimensions.height)
    length = _wall_length_mm(wall)
    thickness = float(wall.thickness_mm or DEFAULT_WALL_THICKNESS_MM)

    if wall.anchor in {"east", "west"}:
        dimensions = {
            "width": thickness,
            "depth": length,
            "height": height,
        }
    else:
        dimensions = {
            "width": length,
            "depth": thickness,
            "height": height,
        }

    return {
        "is_wall": True,
        "position_mm": {
            "x": (min_x + max_x) / 2.0,
            "y": (min_y + max_y) / 2.0,
            "z": height / 2.0,
        },
        "dimensions_mm": dimensions,
        "rotation_z_deg": 0.0,
    }


def _opening_layout_item(opening: Opening) -> dict[str, object]:
    payload = opening.model_dump(mode="json")
    return {
        "is_door": opening.kind == "door",
        "is_window": opening.kind == "window",
        "anchor_wall": opening.wall,
        "position_mm": payload["center_mm"],
        "dimensions_mm": payload["dimensions_mm"],
        "rotation_z_deg": 0.0,
    }


def _structural_layout(environment: Environment) -> dict[str, dict[str, object]]:
    layout: dict[str, dict[str, object]] = {}
    for wall in environment.wall:
        layout[wall.name] = _wall_layout_item(wall)
    for opening in environment.openings:
        layout[opening.id] = _opening_layout_item(opening)
    return layout


def _placed_layout_item(item: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"is_wall": False}
    for key in (
        "product_id",
        "position_mm",
        "dimensions_mm",
        "rotation_z_deg",
        "anchor_wall",
        "zone_type",
    ):
        if key in item:
            payload[key] = item[key]
    return payload


def _variant_layout(
    environment: Environment,
    variant: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    layout = _structural_layout(environment)
    product_layout = _as_mapping(variant.get("layout"))
    for key in sorted(product_layout):
        item = _as_mapping(product_layout[key])
        if item:
            layout[str(key)] = _placed_layout_item(item)
    return layout


def _family_label(variant: Mapping[str, object]) -> str:
    if variant.get("family_label"):
        return str(variant["family_label"])
    labels = {"I": "I-shaped", "L": "L-shaped", "U": "U-shaped"}
    return labels.get(str(variant.get("family")), str(variant.get("family") or "layout"))


def _violations(variant: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            key: violation[key]
            for key in ("rule_id", "text", "severity")
            if key in violation
        }
        for violation in (_as_mapping(raw) for raw in _as_list(variant.get("violations")))
        if violation
    ]


def _rationale(variant: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "rule_id": str(rationale.get("rule_id") or ""),
            "text": str(rationale.get("text") or ""),
        }
        for rationale in (_as_mapping(raw) for raw in _as_list(variant.get("rationale")))
        if rationale.get("rule_id") and rationale.get("text")
    ]


def _layout_variant(
    design_input: DesignInput,
    variant: Mapping[str, object],
) -> dict[str, object]:
    return {
        "id": str(variant.get("id") or "variant"),
        "family": _family_label(variant),
        "score": round(_as_float(variant.get("score")), 4),
        "violations": _violations(variant),
        "environment": design_input.environment.model_dump(mode="json"),
        "layout": _variant_layout(design_input.environment, variant),
        "rationale": _rationale(variant),
    }


def _request_fingerprint(
    design_input: DesignInput,
    layouts: list[dict[str, object]],
) -> str:
    payload = {
        "environment": design_input.environment.model_dump(mode="json"),
        "preferences": design_input.preferences.model_dump(mode="json"),
        "layouts": [
            {
                "id": layout["id"],
                "family": layout["family"],
                "score": layout["score"],
                "violations": layout["violations"],
                "layout": layout["layout"],
                "rationale": layout["rationale"],
            }
            for layout in layouts
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _deterministic_request_id(
    design_input: DesignInput,
    layouts: list[dict[str, object]],
) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"kdu-cyncly-auto-design:{_request_fingerprint(design_input, layouts)}",
        )
    )


def _deterministic_duration_ms(
    variants: list[dict[str, object]],
    repairs: list[dict[str, object]],
    violations: list[dict[str, object]],
) -> int:
    return 1000 + (125 * len(variants)) + (10 * len(repairs)) + (15 * len(violations))


def build_renderer_output_envelope(
    *,
    design_input: DesignInput,
    variants: list[dict[str, object]],
    repairs: list[dict[str, object]] | None = None,
    violations: list[dict[str, object]] | None = None,
    request_id: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, object]:
    """Convert internal ranked variants to the renderer response envelope."""

    layouts = [
        _layout_variant(design_input, variant)
        for variant in variants
    ]
    repair_payloads = repairs or []
    violation_payloads = violations or []
    envelope = {
        "request_id": request_id or _deterministic_request_id(design_input, layouts),
        "duration_ms": (
            duration_ms
            if duration_ms is not None
            else _deterministic_duration_ms(variants, repair_payloads, violation_payloads)
        ),
        "layouts": layouts,
    }
    return LayoutResponse.model_validate(envelope).model_dump(
        mode="json",
        exclude_none=True,
    )
