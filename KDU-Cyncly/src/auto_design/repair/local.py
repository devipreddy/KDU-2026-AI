from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from auto_design.catalog.service import CatalogService
from auto_design.geometry import (
    AABB,
    DimensionsMM,
    PointMM,
    RoomBounds,
    build_opening_constraints,
)
from auto_design.schemas.catalog import Product
from auto_design.schemas.environment import Environment
from auto_design.schemas.intent import StructuredIntent
from auto_design.validation import (
    VariantValidationResult,
    flatten_validation_results,
    validate_variant,
)


BACKED_COMPONENTS = {
    "dishwasher",
    "double_sink",
    "fridge",
    "hood",
    "microwave",
    "oven",
    "single_sink",
    "sink",
    "stove",
}
BASE_COMPONENTS = {"base_cabinet", "prep_base_cabinet"}
MAX_CONTINUITY_GAP_MM = 50.0
OPENING_BUFFER_MM = 50.0


@dataclass(frozen=True)
class RepairAction:
    variant_id: str
    rule_id: str
    action: str
    item_keys: tuple[str, ...]
    text: str

    def to_payload(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "rule_id": self.rule_id,
            "action": self.action,
            "item_keys": list(self.item_keys),
            "text": self.text,
        }


@dataclass(frozen=True)
class RepairResult:
    variant: dict[str, object]
    actions: tuple[RepairAction, ...]
    validation: VariantValidationResult

    def to_payload(self) -> dict[str, object]:
        return {
            "variant_id": str(self.variant.get("id") or ""),
            "actions": [action.to_payload() for action in self.actions],
            "validation": self.validation.to_payload(),
        }


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_mutable_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _json_signature(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _placement(variant: dict[str, object]) -> dict[str, Any]:
    return _as_mutable_mapping(variant.get("placement"))


def _runs(variant: dict[str, object]) -> list[dict[str, Any]]:
    return [
        run for run in _as_list(_placement(variant).get("runs"))
        if isinstance(run, dict)
    ]


def _overhead_items(variant: dict[str, object]) -> list[dict[str, Any]]:
    return [
        item for item in _as_list(_placement(variant).get("overhead_items"))
        if isinstance(item, dict)
    ]


def _overhead_item_store(variant: dict[str, object]) -> list[Any]:
    placement = _placement(variant)
    items = placement.get("overhead_items")
    if not isinstance(items, list):
        items = []
        placement["overhead_items"] = items
    return items


def _run_items(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _as_list(run.get("items")) if isinstance(item, dict)]


def _all_items(variant: dict[str, object]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for run in _runs(variant):
        items.extend(_run_items(run))
    items.extend(_overhead_items(variant))
    return items


def _item_key(item: Mapping[str, object]) -> str:
    return str(item.get("key") or item.get("product_id") or "item")


def _item_width(item: Mapping[str, object]) -> float:
    dimensions = _as_mapping(item.get("dimensions_mm"))
    span_width = _as_float(item.get("end_mm")) - _as_float(item.get("start_mm"))
    return _as_float(
        dimensions.get("width"),
        _as_float(item.get("width_mm"), span_width),
    )


def _item_depth(item: Mapping[str, object]) -> float:
    return _as_float(_as_mapping(item.get("dimensions_mm")).get("depth"))


def _item_height(item: Mapping[str, object]) -> float:
    return _as_float(_as_mapping(item.get("dimensions_mm")).get("height"))


def _item_aabb(item: Mapping[str, object]) -> AABB:
    position = _as_mapping(item.get("position_mm"))
    dimensions = _as_mapping(item.get("dimensions_mm"))
    return AABB.from_center(
        PointMM(
            x=_as_float(position.get("x")),
            y=_as_float(position.get("y")),
            z=_as_float(position.get("z")),
        ),
        DimensionsMM(
            width=_as_float(dimensions.get("width")),
            depth=_as_float(dimensions.get("depth")),
            height=_as_float(dimensions.get("height")),
        ),
        _as_float(item.get("rotation_z_deg")),
    )


def _product_dimensions(product: Product) -> dict[str, float]:
    return {
        "width": float(product.width_mm),
        "depth": float(product.depth_mm),
        "height": float(product.height_mm),
    }


def _preferred_base_skus(intent: StructuredIntent | None) -> set[str]:
    if intent is None or intent.cabinet_color is None:
        return set()
    if intent.cabinet_color.target not in {"cabinets", "base_cabinets"}:
        return set()
    return set(intent.cabinet_color.matched_skus)


def _base_product(
    catalog: CatalogService,
    *,
    min_width_mm: float = 0.0,
    intent: StructuredIntent | None = None,
) -> Product:
    preferred_skus = _preferred_base_skus(intent)
    if preferred_skus:
        preferred = [
            product for product in catalog.by_category("cabinet")
            if product.id in preferred_skus
            and product.type.startswith(("base_", "base"))
            and float(product.width_mm) >= min_width_mm
        ]
        if preferred:
            return min(preferred, key=lambda product: (float(product.width_mm), product.id))

    candidates = [
        product for product in catalog.by_category("cabinet")
        if product.type.startswith(("base_cabinet", "base_drawer"))
        and float(product.depth_mm) == 600.0
        and float(product.width_mm) >= min_width_mm
    ]
    if candidates:
        return min(candidates, key=lambda product: (float(product.width_mm), product.id))

    fallback = [
        product for product in catalog.by_category("cabinet")
        if "base" in product.type
        and float(product.width_mm) >= min_width_mm
    ]
    if fallback:
        return min(fallback, key=lambda product: (float(product.width_mm), product.id))

    return min(
        catalog.by_category("cabinet"),
        key=lambda product: (float(product.width_mm), product.id),
    )


def _product_matching(catalog: CatalogService, *needles: str) -> Product | None:
    lowered = tuple(needle.casefold() for needle in needles)
    matches = [
        product for product in catalog.products
        if any(needle in product.type.casefold() for needle in lowered)
    ]
    if not matches:
        return None
    return min(matches, key=lambda product: (float(product.width_mm), product.id))


def _next_repair_key(variant: dict[str, object], prefix: str) -> str:
    keys = {_item_key(item) for item in _all_items(variant)}
    index = 1
    while f"{prefix}_{index}" in keys:
        index += 1
    return f"{prefix}_{index}"


def _position_item_on_wall(item: dict[str, Any], bounds: RoomBounds) -> None:
    wall = item.get("wall")
    if wall not in {"north", "south", "east", "west"}:
        return
    start = _as_float(item.get("start_mm"))
    end = _as_float(item.get("end_mm"), start + _item_width(item))
    position = _as_mapping(item.get("position_mm"))
    z_center = _as_float(position.get("z"), _item_height(item) / 2.0)
    wall_system = bounds.wall(wall)
    item["position_mm"] = wall_system.center_for(
        offset_mm=(start + end) / 2.0,
        depth_mm=_item_depth(item),
        height_mm=_item_height(item),
        z_center_mm=z_center,
    ).to_dict()
    item["rotation_z_deg"] = wall_system.rotation_z_deg


def _sync_run(run: dict[str, Any], bounds: RoomBounds, *, start_mm: float | None = None) -> None:
    cursor = _as_float(run.get("start_mm")) if start_mm is None else start_mm
    run["start_mm"] = cursor
    items = _run_items(run)
    for sequence_index, item in enumerate(items, start=1):
        width = _item_width(item)
        item["sequence_index"] = sequence_index
        item["start_mm"] = cursor
        item["end_mm"] = cursor + width
        item["width_mm"] = width
        cursor += width
        _position_item_on_wall(item, bounds)
    run["end_mm"] = cursor
    gaps = [
        _as_float(items[index + 1].get("start_mm")) - _as_float(items[index].get("end_mm"))
        for index in range(len(items) - 1)
    ]
    run["continuity_gaps_mm"] = gaps
    run["is_continuous"] = all(abs(gap) <= MAX_CONTINUITY_GAP_MM for gap in gaps)
    run["starts_with_base"] = bool(items and _is_base_item(items[0]))
    run["ends_with_base"] = bool(items and _is_base_item(items[-1]))


def _sync_all_runs(variant: dict[str, object], bounds: RoomBounds) -> None:
    for run in _runs(variant):
        _sync_run(run, bounds)


def _sync_overhead_positions(variant: dict[str, object], bounds: RoomBounds) -> None:
    base_items = {
        _item_key(item): item
        for run in _runs(variant)
        for item in _run_items(run)
    }
    for item in _overhead_items(variant):
        anchor_key = item.get("anchored_to_key")
        anchor = base_items.get(str(anchor_key)) if anchor_key else None
        if anchor is None and item.get("component") == "hood":
            anchor = next(
                (
                    base_item for base_item in base_items.values()
                    if base_item.get("component") == "stove"
                ),
                None,
            )
        if anchor is not None:
            center_offset = (
                _as_float(anchor.get("start_mm"))
                + _as_float(anchor.get("end_mm"))
            ) / 2.0
            width = _item_width(item)
            item["wall"] = anchor.get("wall")
            item["run_role"] = anchor.get("run_role")
            item["start_mm"] = center_offset - (width / 2.0)
            item["end_mm"] = center_offset + (width / 2.0)
            item["anchored_to_key"] = _item_key(anchor)
        _position_item_on_wall(item, bounds)


def _sync_layout(variant: dict[str, object]) -> None:
    layout: dict[str, dict[str, object]] = {}
    for item in _all_items(variant):
        key = _item_key(item)
        layout[key] = {
            "product_id": item.get("product_id"),
            "position_mm": item.get("position_mm"),
            "dimensions_mm": item.get("dimensions_mm"),
            "rotation_z_deg": item.get("rotation_z_deg"),
            "anchor_wall": item.get("wall"),
            "zone_type": item.get("zone_type"),
        }
    variant["layout"] = layout


def _sync_diversity(variant: dict[str, object]) -> None:
    placement = _placement(variant)
    zone_plan = _as_mapping(variant.get("zone_plan"))
    zone_wall_map = {
        str(assignment.get("item")): str(assignment.get("wall"))
        for assignment in _as_list(zone_plan.get("item_assignments"))
        if isinstance(assignment, Mapping)
        and assignment.get("item") in {
            "dishwasher",
            "double_sink",
            "fridge",
            "hood",
            "single_sink",
            "sink",
            "stove",
            "tall_cabinet",
        }
    }
    run_sequences = {
        f"{run.get('wall')}:{run.get('run_role')}": [
            str(item.get("component"))
            for item in _run_items(run)
        ]
        for run in _runs(variant)
    }
    component_offsets = {
        str(item.get("component")): {
            "wall": item.get("wall"),
            "start_mm": item.get("start_mm"),
            "end_mm": item.get("end_mm"),
        }
        for item in _all_items(variant)
        if item.get("component") in {
            "dishwasher",
            "double_sink",
            "fridge",
            "single_sink",
            "sink",
            "stove",
        }
    }
    signature = _json_signature(
        {
            "walls": zone_wall_map,
            "sequences": run_sequences,
            "offsets": component_offsets,
        }
    )
    placement["diversity_signature"] = signature
    variant["diversity"] = {
        "signature": signature,
        "zone_wall_map": zone_wall_map,
        "run_sequences": run_sequences,
        "component_offsets": component_offsets,
    }


def _recompute_summary(variant: dict[str, object]) -> None:
    placement = _placement(variant)
    runs = _runs(variant)
    placement["is_continuous"] = all(bool(run.get("is_continuous")) for run in runs)
    placement["item_count"] = len(_all_items(variant))
    coverages = _as_list(placement.get("base_coverages"))
    covered_keys = {
        str(_as_mapping(coverage).get("covered_item_key"))
        for coverage in coverages
    }
    required_keys = {
        _item_key(item)
        for item in _all_items(variant)
        if item.get("component") in BACKED_COMPONENTS
    }
    placement["base_coverage_valid"] = required_keys.issubset(covered_keys) and all(
        bool(_as_mapping(coverage).get("is_sufficient"))
        for coverage in coverages
    )


def _sync_variant(variant: dict[str, object], bounds: RoomBounds) -> None:
    _sync_overhead_positions(variant, bounds)
    _recompute_summary(variant)
    _sync_layout(variant)
    _sync_diversity(variant)


def _is_base_item(item: Mapping[str, object]) -> bool:
    return bool(item.get("is_base_cabinet")) or item.get("component") in BASE_COMPONENTS


def _new_base_item(
    variant: dict[str, object],
    catalog: CatalogService,
    run: Mapping[str, object],
    *,
    prefix: str,
    terminator: bool,
    min_width_mm: float = 0.0,
    intent: StructuredIntent | None = None,
) -> dict[str, Any]:
    product = _base_product(catalog, min_width_mm=min_width_mm, intent=intent)
    dimensions = _product_dimensions(product)
    return {
        "key": _next_repair_key(variant, prefix),
        "component": "base_cabinet",
        "product_id": product.id,
        "product_type": product.type,
        "wall": run.get("wall"),
        "run_role": run.get("run_role"),
        "zone_type": "storage",
        "layer": "base_run",
        "sequence_index": 0,
        "start_mm": 0.0,
        "end_mm": dimensions["width"],
        "width_mm": dimensions["width"],
        "position_mm": {"x": 0.0, "y": 0.0, "z": dimensions["height"] / 2.0},
        "dimensions_mm": dimensions,
        "rotation_z_deg": 0.0,
        "backed_by_product_id": None,
        "backing_width_mm": None,
        "is_base_cabinet": True,
        "is_terminator": terminator,
    }


def _rebuild_base_coverages(
    variant: dict[str, object],
    catalog: CatalogService,
    bounds: RoomBounds,
    intent: StructuredIntent | None = None,
) -> tuple[str, ...]:
    coverages: list[dict[str, object]] = []
    repaired_item_keys: list[str] = []
    for item in _all_items(variant):
        if item.get("component") not in BACKED_COMPONENTS:
            continue
        width = _item_width(item)
        base = _base_product(catalog, min_width_mm=width, intent=intent)
        dimensions = _product_dimensions(base)
        start = _as_float(item.get("start_mm"))
        end = _as_float(item.get("end_mm"))
        center_offset = (start + end) / 2.0
        coverage_start = center_offset - (dimensions["width"] / 2.0)
        coverage_end = center_offset + (dimensions["width"] / 2.0)
        wall = item.get("wall")
        if wall in {"north", "south", "east", "west"}:
            wall_system = bounds.wall(wall)
            position = wall_system.center_for(
                offset_mm=center_offset,
                depth_mm=dimensions["depth"],
                height_mm=dimensions["height"],
            ).to_dict()
            rotation = wall_system.rotation_z_deg
        else:
            position = {"x": 0.0, "y": 0.0, "z": dimensions["height"] / 2.0}
            rotation = 0.0
        item["backed_by_product_id"] = base.id
        item["backing_width_mm"] = dimensions["width"]
        repaired_item_keys.append(_item_key(item))
        coverages.append(
            {
                "covered_item_key": _item_key(item),
                "covered_component": item.get("component"),
                "covered_product_id": item.get("product_id"),
                "base_product_id": base.id,
                "base_product_type": base.type,
                "wall": item.get("wall"),
                "run_role": item.get("run_role"),
                "zone_type": item.get("zone_type"),
                "kind": "behind" if item.get("layer") == "overhead" else "below",
                "start_mm": coverage_start,
                "end_mm": coverage_end,
                "covered_width_mm": width,
                "coverage_width_mm": dimensions["width"],
                "position_mm": position,
                "dimensions_mm": dimensions,
                "rotation_z_deg": rotation,
                "is_sufficient": dimensions["width"] >= width,
            }
        )
    _placement(variant)["base_coverages"] = coverages
    return tuple(repaired_item_keys)


def _compact_runs_to_shared_corner(
    variant: dict[str, object],
    bounds: RoomBounds,
) -> tuple[str, ...]:
    runs = _runs(variant)
    walls = {str(run.get("wall")) for run in runs}
    changed: list[str] = []
    for run in runs:
        wall = run.get("wall")
        if wall not in {"north", "south", "east", "west"}:
            continue
        run_width = sum(_item_width(item) for item in _run_items(run))
        wall_length = bounds.wall(wall).length_mm
        desired_start: float | None = None
        if wall in {"north", "south"}:
            if "east" in walls:
                desired_start = max(0.0, wall_length - run_width)
            elif "west" in walls:
                desired_start = 0.0
        elif wall in {"east", "west"}:
            if "north" in walls:
                desired_start = max(0.0, wall_length - run_width)
            elif "south" in walls:
                desired_start = 0.0
        if desired_start is None:
            continue
        if abs(_as_float(run.get("start_mm")) - desired_start) <= 1.0:
            continue
        _sync_run(run, bounds, start_mm=desired_start)
        changed.append(str(wall))
    return tuple(changed)


def _repair_terminators_and_continuity(
    variant: dict[str, object],
    catalog: CatalogService,
    bounds: RoomBounds,
    intent: StructuredIntent | None = None,
) -> tuple[str, ...]:
    changed: list[str] = []
    for run in _runs(variant):
        items = _run_items(run)
        if not items:
            continue
        if not _is_base_item(items[0]):
            new_item = _new_base_item(
                variant,
                catalog,
                run,
                prefix=f"repair_start_base_{run.get('wall')}",
                terminator=True,
                intent=intent,
            )
            items.insert(0, new_item)
            run["items"] = items
            changed.append(_item_key(new_item))
        if not _is_base_item(items[-1]):
            new_item = _new_base_item(
                variant,
                catalog,
                run,
                prefix=f"repair_end_base_{run.get('wall')}",
                terminator=True,
                intent=intent,
            )
            items.append(new_item)
            run["items"] = items
            changed.append(_item_key(new_item))

        gaps = [
            _as_float(items[index + 1].get("start_mm"))
            - _as_float(items[index].get("end_mm"))
            for index in range(len(items) - 1)
        ]
        base_width = float(_base_product(catalog, intent=intent).width_mm)
        insert_after: list[int] = [
            index for index, gap in enumerate(gaps)
            if abs(gap) > MAX_CONTINUITY_GAP_MM and gap >= base_width
        ]
        for index in reversed(insert_after):
            new_item = _new_base_item(
                variant,
                catalog,
                run,
                prefix=f"repair_filler_base_{run.get('wall')}",
                terminator=False,
                min_width_mm=base_width,
                intent=intent,
            )
            items.insert(index + 1, new_item)
            changed.append(_item_key(new_item))
        if insert_after:
            run["items"] = items
        old_gaps = tuple(_as_list(run.get("continuity_gaps_mm")))
        _sync_run(run, bounds)
        if old_gaps != tuple(_as_list(run.get("continuity_gaps_mm"))):
            changed.append(f"{run.get('wall')}_run")
    return tuple(dict.fromkeys(changed))


def _realign_or_add_hood(
    variant: dict[str, object],
    catalog: CatalogService,
    bounds: RoomBounds,
) -> tuple[str, ...]:
    stove = next((item for item in _all_items(variant) if item.get("component") == "stove"), None)
    if stove is None:
        return ()
    hoods = [item for item in _overhead_items(variant) if item.get("component") == "hood"]
    if not hoods:
        product = _product_matching(catalog, "hood")
        if product is None:
            return ()
        dimensions = _product_dimensions(product)
        hood = {
            "key": _next_repair_key(variant, "repair_hood"),
            "component": "hood",
            "product_id": product.id,
            "product_type": product.type,
            "wall": stove.get("wall"),
            "run_role": stove.get("run_role"),
            "zone_type": "cooking",
            "layer": "overhead",
            "sequence_index": stove.get("sequence_index"),
            "start_mm": stove.get("start_mm"),
            "end_mm": stove.get("end_mm"),
            "width_mm": dimensions["width"],
            "position_mm": {"x": 0.0, "y": 0.0, "z": 1900.0},
            "dimensions_mm": dimensions,
            "rotation_z_deg": stove.get("rotation_z_deg", 0.0),
            "backed_by_product_id": None,
            "backing_width_mm": None,
            "is_base_cabinet": False,
            "is_terminator": False,
            "anchored_to_key": _item_key(stove),
        }
        _overhead_item_store(variant).append(hood)
        _sync_overhead_positions(variant, bounds)
        return (_item_key(hood),)

    changed: list[str] = []
    for hood in hoods:
        before = dict(_as_mapping(hood.get("position_mm")))
        hood["anchored_to_key"] = _item_key(stove)
        _sync_overhead_positions(variant, bounds)
        after = dict(_as_mapping(hood.get("position_mm")))
        if before != after:
            changed.append(_item_key(hood))
    return tuple(changed)


def _move_component_next_to_anchor(
    variant: dict[str, object],
    component: str,
    anchor_component: str,
    *,
    place_after: bool,
    bounds: RoomBounds,
) -> tuple[str, ...]:
    for run in _runs(variant):
        items = _run_items(run)
        moved = next((item for item in items if item.get("component") == component), None)
        anchor = next(
            (item for item in items if item.get("component") == anchor_component),
            None,
        )
        if moved is None or anchor is None:
            continue
        items.remove(moved)
        anchor_index = items.index(anchor)
        insert_at = anchor_index + 1 if place_after else anchor_index
        items.insert(insert_at, moved)
        run["items"] = items
        _sync_run(run, bounds)
        return (_item_key(moved), _item_key(anchor))
    return ()


def _move_fridge_to_corner(variant: dict[str, object], bounds: RoomBounds) -> tuple[str, ...]:
    for run in _runs(variant):
        items = _run_items(run)
        fridge = next((item for item in items if item.get("component") == "fridge"), None)
        if fridge is None:
            continue
        items.remove(fridge)
        first_base_index = next(
            (index for index, item in enumerate(items) if _is_base_item(item)),
            -1,
        )
        items.insert(max(first_base_index + 1, 0), fridge)
        run["items"] = items
        _sync_run(run, bounds)
        return (_item_key(fridge),)
    return ()


def _run_for_item(
    variant: dict[str, object],
    target: Mapping[str, object],
) -> dict[str, Any] | None:
    target_key = _item_key(target)
    for run in _runs(variant):
        if any(_item_key(item) == target_key for item in _run_items(run)):
            return run
    return None


def _separate_stove_and_fridge(
    variant: dict[str, object],
    catalog: CatalogService,
    bounds: RoomBounds,
    intent: StructuredIntent | None = None,
) -> tuple[str, ...]:
    stove = next((item for item in _all_items(variant) if item.get("component") == "stove"), None)
    fridge = next((item for item in _all_items(variant) if item.get("component") == "fridge"), None)
    if stove is None or fridge is None:
        return ()

    stove_run = _run_for_item(variant, stove)
    fridge_run = _run_for_item(variant, fridge)
    if stove_run is None or fridge_run is None:
        return ()

    if stove_run is fridge_run:
        items = _run_items(stove_run)
        stove_index = next(index for index, item in enumerate(items) if item is stove)
        fridge_index = next(index for index, item in enumerate(items) if item is fridge)
        if abs(stove_index - fridge_index) == 1:
            insert_at = max(stove_index, fridge_index)
            separator = _new_base_item(
                variant,
                catalog,
                stove_run,
                prefix=f"repair_separator_base_{stove_run.get('wall')}",
                terminator=False,
                min_width_mm=600.0,
                intent=intent,
            )
            items.insert(insert_at, separator)
            stove_run["items"] = items
            run_width = sum(_item_width(item) for item in items)
            wall = stove_run.get("wall")
            start = _as_float(stove_run.get("start_mm"))
            if wall in {"north", "south", "east", "west"}:
                start = min(start, max(0.0, bounds.wall(wall).length_mm - run_width))
            _sync_run(stove_run, bounds, start_mm=start)
            return (_item_key(stove), _item_key(fridge), _item_key(separator))
        _sync_run(stove_run, bounds)
        return (_item_key(stove), _item_key(fridge))

    run_to_shift = stove_run
    wall = run_to_shift.get("wall")
    if wall not in {"north", "south", "east", "west"}:
        return ()
    run_width = sum(_item_width(item) for item in _run_items(run_to_shift))
    wall_length = bounds.wall(wall).length_mm
    current_start = _as_float(run_to_shift.get("start_mm"))
    if current_start > wall_length / 2.0:
        desired_start = max(0.0, current_start - 900.0)
    else:
        desired_start = min(wall_length - run_width, current_start + 900.0)
    _sync_run(run_to_shift, bounds, start_mm=max(0.0, desired_start))
    return (_item_key(stove), _item_key(fridge))


def _move_stove_away_from_fridge(variant: dict[str, object], bounds: RoomBounds) -> tuple[str, ...]:
    for run in _runs(variant):
        items = _run_items(run)
        stove = next((item for item in items if item.get("component") == "stove"), None)
        if stove is None:
            continue
        items.remove(stove)
        insert_at = max(len(items) - 1, 0)
        items.insert(insert_at, stove)
        run["items"] = items
        _sync_run(run, bounds)
        return (_item_key(stove),)
    return ()


def _shift_runs_away_from_openings(
    environment: Environment,
    variant: dict[str, object],
    bounds: RoomBounds,
) -> tuple[str, ...]:
    constraints = build_opening_constraints(environment)
    changed: list[str] = []
    for run in _runs(variant):
        wall = run.get("wall")
        if wall not in {"north", "south", "east", "west"}:
            continue
        spans = constraints.blocked_spans_for_wall(wall)
        if not spans:
            continue
        run_width = sum(_item_width(item) for item in _run_items(run))
        wall_length = bounds.wall(wall).length_mm
        candidates = [0.0]
        for span in spans:
            candidates.append(max(0.0, span.end_mm + OPENING_BUFFER_MM))
            candidates.append(max(0.0, span.start_mm - run_width - OPENING_BUFFER_MM))
        current_start = _as_float(run.get("start_mm"))
        ordered_candidates = sorted(
            set(candidates),
            key=lambda value: abs(value - current_start),
        )
        for start in ordered_candidates:
            if start + run_width > wall_length:
                continue
            end = start + run_width
            if any(span.overlaps(start, end) for span in spans):
                continue
            _sync_run(run, bounds, start_mm=start)
            changed.append(f"{wall}_run")
            break

    reservations = constraints.door_swing_reservations
    for item in _all_items(variant):
        if any(_item_aabb(item).intersects(reservation) for reservation in reservations):
            _position_item_on_wall(item, bounds)
            if not any(_item_aabb(item).intersects(reservation) for reservation in reservations):
                changed.append(_item_key(item))
    return tuple(dict.fromkeys(changed))


def _repair_once(
    environment: Environment,
    catalog: CatalogService,
    variant: dict[str, object],
    validation: VariantValidationResult,
    actions: list[RepairAction],
    intent: StructuredIntent | None = None,
) -> bool:
    bounds = RoomBounds.from_environment(environment)
    variant_id = str(variant.get("id") or "variant")
    rule_ids = {violation.rule_id for violation in validation.violations}
    changed = False

    if {"LAYOUT-03", "LAYOUT-05"} & rule_ids:
        item_keys = _repair_terminators_and_continuity(
            variant,
            catalog,
            bounds,
            intent,
        )
        if item_keys:
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id="LAYOUT-03",
                    action="insert_base_and_fix_continuity",
                    item_keys=item_keys,
                    text="Inserted base fillers or terminators and snapped the run.",
                )
            )
            changed = True

    if "WORKFLOW-03" in rule_ids:
        walls = _compact_runs_to_shared_corner(variant, bounds)
        if walls:
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id="WORKFLOW-03",
                    action="compact_runs_to_shared_corner",
                    item_keys=walls,
                    text="Shifted active wall runs toward their shared corner.",
                )
            )
            changed = True

    if "WORKFLOW-01" in rule_ids:
        item_keys = _move_component_next_to_anchor(
            variant,
            "dishwasher",
            "sink",
            place_after=True,
            bounds=bounds,
        )
        if item_keys:
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id="WORKFLOW-01",
                    action="move_dishwasher_next_to_sink",
                    item_keys=item_keys,
                    text="Moved dishwasher adjacent to the sink on its run.",
                )
            )
            changed = True

    if {"NKBA-CL-01", "LAYOUT-06"} & rule_ids:
        item_keys = _move_fridge_to_corner(variant, bounds)
        if item_keys:
            rule_id = "NKBA-CL-01" if "NKBA-CL-01" in rule_ids else "LAYOUT-06"
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id=rule_id,
                    action="shift_fridge_to_corner",
                    item_keys=item_keys,
                    text="Moved refrigerator back near the run corner.",
                )
            )
            changed = True

    if "WORKFLOW-02" in rule_ids:
        item_keys = _separate_stove_and_fridge(variant, catalog, bounds, intent)
        if not item_keys:
            item_keys = _move_stove_away_from_fridge(variant, bounds)
        if item_keys:
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id="WORKFLOW-02",
                    action="separate_stove_and_fridge",
                    item_keys=item_keys,
                    text="Separated stove and fridge with distance or base filler.",
                )
            )
            changed = True

    if "LAYOUT-02" in rule_ids:
        item_keys = _realign_or_add_hood(variant, catalog, bounds)
        if item_keys:
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id="LAYOUT-02",
                    action="realign_hood_over_stove",
                    item_keys=item_keys,
                    text="Centered the hood over the stove.",
                )
            )
            changed = True

    if "NKBA-CL-02" in rule_ids:
        item_keys = _shift_runs_away_from_openings(environment, variant, bounds)
        if item_keys:
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id="NKBA-CL-02",
                    action="avoid_door_and_window_openings",
                    item_keys=item_keys,
                    text="Moved blocked items or runs away from opening reservations.",
                )
            )
            changed = True

    if "LAYOUT-04" in rule_ids:
        item_keys = _rebuild_base_coverages(variant, catalog, bounds, intent)
        if item_keys:
            actions.append(
                RepairAction(
                    variant_id=variant_id,
                    rule_id="LAYOUT-04",
                    action="rebuild_base_coverage",
                    item_keys=item_keys,
                    text="Rebuilt equal-or-larger base backing records.",
                )
            )
            changed = True

    if changed:
        _sync_all_runs(variant, bounds)
        _sync_variant(variant, bounds)
    return changed


def repair_variant(
    environment: Environment,
    catalog: CatalogService,
    variant: Mapping[str, object],
    *,
    intent: StructuredIntent | None = None,
    max_passes: int = 3,
) -> RepairResult:
    repaired = copy.deepcopy(dict(variant))
    actions: list[RepairAction] = []
    validation = validate_variant(environment, repaired)

    for _ in range(max_passes):
        if not validation.violations:
            break
        changed = _repair_once(
            environment,
            catalog,
            repaired,
            validation,
            actions,
            intent,
        )
        validation = validate_variant(environment, repaired)
        if not changed:
            break

    repaired["violations"] = [
        violation.to_payload()
        for violation in validation.violations
    ]
    repaired["repair_history"] = [action.to_payload() for action in actions]
    return RepairResult(
        variant=repaired,
        actions=tuple(actions),
        validation=validation,
    )


def repair_variants(
    environment: Environment,
    catalog: CatalogService,
    variants: list[dict[str, object]],
    intent: StructuredIntent | None = None,
) -> tuple[RepairResult, ...]:
    return tuple(
        repair_variant(environment, catalog, variant, intent=intent)
        for variant in variants
    )


def flatten_repair_actions(results: tuple[RepairResult, ...]) -> list[dict[str, object]]:
    return [
        action.to_payload()
        for result in results
        for action in result.actions
    ]


def flatten_repair_violations(results: tuple[RepairResult, ...]) -> list[dict[str, object]]:
    return flatten_validation_results(tuple(result.validation for result in results))
