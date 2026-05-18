from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from auto_design.geometry import (
    AABB,
    DimensionsMM,
    PointMM,
    RoomBounds,
    build_opening_constraints,
)
from auto_design.schemas.environment import Environment, Opening, WallAnchor


ViolationSeverity = Literal["hard", "soft", "warning", "error"]

FRIDGE_FRONT_CLEARANCE_MM = 1067.0
DOOR_SWING_CLEARANCE_MM = 900.0
DISHWASHER_SINK_MAX_GAP_MM = 600.0
STOVE_FRIDGE_MIN_GAP_MM = 600.0
WORK_TRIANGLE_MIN_MM = 3600.0
WORK_TRIANGLE_MAX_MM = 6600.0
WINDOW_SINK_ALIGNMENT_TOLERANCE_MM = 300.0
HOOD_STOVE_ALIGNMENT_TOLERANCE_MM = 100.0
MAX_CONTINUITY_GAP_MM = 50.0
CORNER_PLACEMENT_TOLERANCE_MM = 900.0

SINK_COMPONENTS = {"sink", "single_sink", "double_sink"}
STOVE_COMPONENTS = {"stove", "oven"}
FRIDGE_COMPONENTS = {"fridge"}
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
STANDALONE_STORAGE_COMPONENTS = {"wall_cabinet", "tall_cabinet"}


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    severity: ViolationSeverity
    text: str

    def to_payload(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "text": self.text,
        }


@dataclass(frozen=True)
class VariantValidationResult:
    variant_id: str
    violations: tuple[RuleViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_payload(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "passed": self.passed,
            "violations": [violation.to_payload() for violation in self.violations],
        }


@dataclass(frozen=True)
class ItemView:
    key: str
    component: str
    product_id: str
    wall: WallAnchor | None
    zone_type: str | None
    layer: str | None
    start_mm: float | None
    end_mm: float | None
    position_mm: PointMM
    dimensions_mm: DimensionsMM
    rotation_z_deg: float
    backed_by_product_id: str | None
    backing_width_mm: float | None
    is_base_cabinet: bool

    @property
    def center_xy(self) -> tuple[float, float]:
        return (self.position_mm.x, self.position_mm.y)

    @property
    def width_mm(self) -> float:
        if self.start_mm is not None and self.end_mm is not None:
            return self.end_mm - self.start_mm
        return self.dimensions_mm.width

    @property
    def bounds(self) -> AABB:
        return AABB.from_center(
            self.position_mm,
            self.dimensions_mm,
            self.rotation_z_deg,
        )


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _point_from_payload(payload: Mapping[str, object]) -> PointMM:
    return PointMM(
        x=_as_float(payload.get("x")),
        y=_as_float(payload.get("y")),
        z=_as_float(payload.get("z")),
    )


def _dimensions_from_payload(payload: Mapping[str, object]) -> DimensionsMM:
    return DimensionsMM(
        width=_as_float(payload.get("width")),
        depth=_as_float(payload.get("depth")),
        height=_as_float(payload.get("height")),
    )


def _item_from_payload(raw_item: Mapping[str, object], *, key: str | None = None) -> ItemView:
    wall = raw_item.get("wall") or raw_item.get("anchor_wall")
    return ItemView(
        key=str(key or raw_item.get("key") or raw_item.get("product_id") or "item"),
        component=str(raw_item.get("component") or raw_item.get("zone_type") or "item"),
        product_id=str(raw_item.get("product_id") or ""),
        wall=wall if wall in {"north", "south", "east", "west"} else None,
        zone_type=str(raw_item.get("zone_type")) if raw_item.get("zone_type") else None,
        layer=str(raw_item.get("layer")) if raw_item.get("layer") else None,
        start_mm=(
            float(raw_item["start_mm"])
            if isinstance(raw_item.get("start_mm"), int | float)
            else None
        ),
        end_mm=(
            float(raw_item["end_mm"])
            if isinstance(raw_item.get("end_mm"), int | float)
            else None
        ),
        position_mm=_point_from_payload(_as_mapping(raw_item.get("position_mm"))),
        dimensions_mm=_dimensions_from_payload(_as_mapping(raw_item.get("dimensions_mm"))),
        rotation_z_deg=_as_float(raw_item.get("rotation_z_deg")),
        backed_by_product_id=(
            str(raw_item.get("backed_by_product_id"))
            if raw_item.get("backed_by_product_id")
            else None
        ),
        backing_width_mm=(
            float(raw_item["backing_width_mm"])
            if isinstance(raw_item.get("backing_width_mm"), int | float)
            else None
        ),
        is_base_cabinet=bool(raw_item.get("is_base_cabinet")),
    )


def _items_from_variant(variant: Mapping[str, object]) -> tuple[ItemView, ...]:
    placement = _as_mapping(variant.get("placement"))
    items: list[ItemView] = []
    for raw_run in _as_list(placement.get("runs")):
        run = _as_mapping(raw_run)
        for raw_item in _as_list(run.get("items")):
            item = _as_mapping(raw_item)
            if item:
                items.append(_item_from_payload(item))
    for raw_item in _as_list(placement.get("overhead_items")):
        item = _as_mapping(raw_item)
        if item:
            items.append(_item_from_payload(item))
    if items:
        return tuple(items)

    layout = _as_mapping(variant.get("layout"))
    return tuple(
        _item_from_payload(_as_mapping(raw_item), key=str(key))
        for key, raw_item in layout.items()
        if isinstance(raw_item, Mapping)
    )


def _runs_from_variant(variant: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    placement = _as_mapping(variant.get("placement"))
    return tuple(
        run for run in (_as_mapping(raw_run) for raw_run in _as_list(placement.get("runs")))
        if run
    )


def _base_coverages_from_variant(variant: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    placement = _as_mapping(variant.get("placement"))
    return tuple(
        coverage
        for coverage in (
            _as_mapping(raw_coverage)
            for raw_coverage in _as_list(placement.get("base_coverages"))
        )
        if coverage
    )


def _first_component(items: tuple[ItemView, ...], components: set[str]) -> ItemView | None:
    return next((item for item in items if item.component in components), None)


def _all_components(items: tuple[ItemView, ...], components: set[str]) -> tuple[ItemView, ...]:
    return tuple(item for item in items if item.component in components)


def _xy_distance(first: ItemView, second: ItemView) -> float:
    return math.dist(first.center_xy, second.center_xy)


def _xy_gap(first: ItemView, second: ItemView) -> float:
    first_bounds = first.bounds
    second_bounds = second.bounds
    dx = max(first_bounds.min_x - second_bounds.max_x, second_bounds.min_x - first_bounds.max_x, 0)
    dy = max(first_bounds.min_y - second_bounds.max_y, second_bounds.min_y - first_bounds.max_y, 0)
    return math.hypot(dx, dy)


def _front_clearance(item: ItemView, bounds: RoomBounds) -> float:
    if item.wall == "north":
        front_face = item.position_mm.y - (item.dimensions_mm.depth / 2.0)
        return front_face - bounds.min_y
    if item.wall == "south":
        front_face = item.position_mm.y + (item.dimensions_mm.depth / 2.0)
        return bounds.max_y - front_face
    if item.wall == "east":
        front_face = item.position_mm.x - (item.dimensions_mm.depth / 2.0)
        return front_face - bounds.min_x
    if item.wall == "west":
        front_face = item.position_mm.x + (item.dimensions_mm.depth / 2.0)
        return bounds.max_x - front_face
    return 0.0


def _window_center_offset(opening: Opening) -> float:
    return float(opening.offset_mm + (opening.width_mm / 2.0))


def _item_center_offset(item: ItemView) -> float | None:
    if item.start_mm is not None and item.end_mm is not None:
        return (item.start_mm + item.end_mm) / 2.0
    return None


def _validate_nkba_cl_01(
    items: tuple[ItemView, ...],
    bounds: RoomBounds,
) -> tuple[RuleViolation, ...]:
    violations: list[RuleViolation] = []
    for fridge in _all_components(items, FRIDGE_COMPONENTS):
        clearance = _front_clearance(fridge, bounds)
        if clearance < FRIDGE_FRONT_CLEARANCE_MM:
            violations.append(
                RuleViolation(
                    rule_id="NKBA-CL-01",
                    severity="hard",
                    text=(
                        f"{fridge.key} has {clearance:.0f} mm clear in front; "
                        f"{FRIDGE_FRONT_CLEARANCE_MM:.0f} mm is required."
                    ),
                )
            )
    return tuple(violations)


def _validate_nkba_cl_02(
    environment: Environment,
    items: tuple[ItemView, ...],
) -> tuple[RuleViolation, ...]:
    constraints = build_opening_constraints(environment)
    violations: list[RuleViolation] = []
    for reservation in constraints.door_swing_reservations:
        blockers = [
            item.key
            for item in items
            if item.bounds.intersects(reservation)
        ]
        if blockers:
            violations.append(
                RuleViolation(
                    rule_id="NKBA-CL-02",
                    severity="hard",
                    text=(
                        "Door swing reservation is blocked by "
                        f"{', '.join(blockers)}; keep a "
                        f"{DOOR_SWING_CLEARANCE_MM:.0f} x "
                        f"{DOOR_SWING_CLEARANCE_MM:.0f} mm clear area."
                    ),
                )
            )
    return tuple(violations)


def _validate_workflow_01(items: tuple[ItemView, ...]) -> tuple[RuleViolation, ...]:
    sink = _first_component(items, SINK_COMPONENTS)
    dishwasher = _first_component(items, {"dishwasher"})
    if sink is None or dishwasher is None:
        return ()
    gap = _xy_gap(sink, dishwasher)
    if gap <= DISHWASHER_SINK_MAX_GAP_MM:
        return ()
    return (
        RuleViolation(
            rule_id="WORKFLOW-01",
            severity="soft",
            text=(
                f"Dishwasher is {gap:.0f} mm from sink; keep it within "
                f"{DISHWASHER_SINK_MAX_GAP_MM:.0f} mm."
            ),
        ),
    )


def _validate_workflow_02(items: tuple[ItemView, ...]) -> tuple[RuleViolation, ...]:
    stove = _first_component(items, {"stove"})
    fridge = _first_component(items, FRIDGE_COMPONENTS)
    if stove is None or fridge is None:
        return ()
    gap = _xy_gap(stove, fridge)
    if gap >= STOVE_FRIDGE_MIN_GAP_MM:
        return ()
    return (
        RuleViolation(
            rule_id="WORKFLOW-02",
            severity="soft",
            text=(
                f"Stove and fridge have {gap:.0f} mm separation; "
                f"{STOVE_FRIDGE_MIN_GAP_MM:.0f} mm is required."
            ),
        ),
    )


def _validate_workflow_03(items: tuple[ItemView, ...]) -> tuple[RuleViolation, ...]:
    sink = _first_component(items, SINK_COMPONENTS)
    stove = _first_component(items, {"stove"})
    fridge = _first_component(items, FRIDGE_COMPONENTS)
    if sink is None or stove is None or fridge is None:
        return ()
    perimeter = _xy_distance(sink, stove) + _xy_distance(stove, fridge) + _xy_distance(fridge, sink)
    if WORK_TRIANGLE_MIN_MM <= perimeter <= WORK_TRIANGLE_MAX_MM:
        return ()
    return (
        RuleViolation(
            rule_id="WORKFLOW-03",
            severity="soft",
            text=(
                f"Work triangle perimeter is {perimeter:.0f} mm; target range is "
                f"{WORK_TRIANGLE_MIN_MM:.0f}-{WORK_TRIANGLE_MAX_MM:.0f} mm."
            ),
        ),
    )


def _validate_layout_01(
    environment: Environment,
    items: tuple[ItemView, ...],
) -> tuple[RuleViolation, ...]:
    windows = [opening for opening in environment.openings if opening.kind == "window"]
    if not windows:
        return ()
    sink = _first_component(items, SINK_COMPONENTS)
    if sink is None or sink.wall is None:
        return (
            RuleViolation(
                rule_id="LAYOUT-01",
                severity="warning",
                text="Window exists, but no sink placement was found for alignment.",
            ),
        )
    sink_offset = _item_center_offset(sink)
    if sink_offset is None:
        return ()
    aligned = any(
        window.wall == sink.wall
        and abs(_window_center_offset(window) - sink_offset)
        <= WINDOW_SINK_ALIGNMENT_TOLERANCE_MM
        for window in windows
    )
    if aligned:
        return ()
    return (
        RuleViolation(
            rule_id="LAYOUT-01",
            severity="warning",
            text=(
                f"{sink.key} is not centered within "
                f"{WINDOW_SINK_ALIGNMENT_TOLERANCE_MM:.0f} mm under any window."
            ),
        ),
    )


def _validate_layout_02(items: tuple[ItemView, ...]) -> tuple[RuleViolation, ...]:
    stove = _first_component(items, {"stove"})
    hood = _first_component(items, {"hood"})
    if stove is None:
        return ()
    if hood is None:
        return (
            RuleViolation(
                rule_id="LAYOUT-02",
                severity="warning",
                text="Stove is placed but no hood was found above it.",
            ),
        )
    offset = _xy_distance(stove, hood)
    if offset <= HOOD_STOVE_ALIGNMENT_TOLERANCE_MM:
        return ()
    return (
        RuleViolation(
            rule_id="LAYOUT-02",
            severity="warning",
            text=(
                f"Hood is {offset:.0f} mm from stove center; maximum allowed "
                f"offset is {HOOD_STOVE_ALIGNMENT_TOLERANCE_MM:.0f} mm."
            ),
        ),
    )


def _validate_layout_03(variant: Mapping[str, object]) -> tuple[RuleViolation, ...]:
    violations: list[RuleViolation] = []
    placement = _as_mapping(variant.get("placement"))
    if placement.get("is_continuous") is False:
        violations.append(
            RuleViolation(
                rule_id="LAYOUT-03",
                severity="hard",
                text=(
                    "Placement payload reports a broken continuous run; "
                    f"maximum gap is {MAX_CONTINUITY_GAP_MM:.0f} mm."
                ),
            )
        )
    for run in _runs_from_variant(variant):
        wall = str(run.get("wall") or "unknown")
        raw_gaps = run.get("continuity_gaps_mm")
        gaps = [
            _as_float(gap)
            for gap in _as_list(raw_gaps)
            if isinstance(gap, int | float)
        ]
        if not gaps:
            items = [
                _item_from_payload(_as_mapping(raw_item))
                for raw_item in _as_list(run.get("items"))
                if isinstance(raw_item, Mapping)
            ]
            gaps = [
                items[index + 1].start_mm - items[index].end_mm
                for index in range(len(items) - 1)
                if items[index].end_mm is not None
                and items[index + 1].start_mm is not None
            ]
        bad_gaps = [gap for gap in gaps if abs(gap) > MAX_CONTINUITY_GAP_MM]
        if run.get("is_continuous") is False and not bad_gaps:
            bad_gaps = [MAX_CONTINUITY_GAP_MM + 1.0]
        if bad_gaps:
            violations.append(
                RuleViolation(
                    rule_id="LAYOUT-03",
                    severity="hard",
                    text=(
                        f"{wall} run has cabinet/appliance gaps "
                        f"{[round(gap, 1) for gap in bad_gaps]}; maximum is "
                        f"{MAX_CONTINUITY_GAP_MM:.0f} mm."
                    ),
                )
            )
    return tuple(violations)


def _validate_layout_04(variant: Mapping[str, object]) -> tuple[RuleViolation, ...]:
    placement = _as_mapping(variant.get("placement"))
    coverage_valid = placement.get("base_coverage_valid")
    coverages = _base_coverages_from_variant(variant)
    covered = {str(coverage.get("covered_item_key")) for coverage in coverages}
    items = _items_from_variant(variant)
    missing = [
        item.key
        for item in items
        if item.component in BACKED_COMPONENTS
        and item.key not in covered
    ]
    insufficient = [
        str(coverage.get("covered_item_key"))
        for coverage in coverages
        if coverage.get("is_sufficient") is False
        or _as_float(coverage.get("coverage_width_mm"))
        < _as_float(coverage.get("covered_width_mm"))
    ]
    if coverage_valid is True and not missing and not insufficient:
        return ()
    if coverage_valid is not False and not missing and not insufficient and coverages:
        return ()
    return (
        RuleViolation(
            rule_id="LAYOUT-04",
            severity="hard",
            text=(
                "Base coverage failed for appliances/sinks; "
                f"missing={missing}, insufficient={insufficient}."
            ),
        ),
    )


def _validate_layout_05(variant: Mapping[str, object]) -> tuple[RuleViolation, ...]:
    violations: list[RuleViolation] = []
    runs = _runs_from_variant(variant)
    base_run_keys: set[tuple[str, str]] = set()
    for run in runs:
        wall = str(run.get("wall") or "unknown")
        run_role = str(run.get("run_role") or "")
        starts_with_base = bool(run.get("starts_with_base"))
        ends_with_base = bool(run.get("ends_with_base"))
        if not starts_with_base or not ends_with_base:
            violations.append(
                RuleViolation(
                    rule_id="LAYOUT-05",
                    severity="hard",
                    text=(
                        f"{wall} run must start and end with base/corner support; "
                        f"starts_with_base={starts_with_base}, "
                        f"ends_with_base={ends_with_base}."
                    ),
                )
            )
        items = [
            _item_from_payload(_as_mapping(raw_item))
            for raw_item in _as_list(run.get("items"))
            if isinstance(raw_item, Mapping)
        ]
        has_base = any(
            item.component in BASE_COMPONENTS or item.is_base_cabinet
            for item in items
        )
        if has_base:
            base_run_keys.add((wall, run_role))
        unsupported = [
            item.key
            for item in items
            if item.component in STANDALONE_STORAGE_COMPONENTS and not has_base
        ]
        if unsupported:
            violations.append(
                RuleViolation(
                    rule_id="LAYOUT-05",
                    severity="hard",
                    text=f"Storage items stand without base support: {', '.join(unsupported)}.",
                )
            )
    for overhead_item in _as_list(_as_mapping(variant.get("placement")).get("overhead_items")):
        item = _item_from_payload(_as_mapping(overhead_item))
        if item.component != "wall_cabinet" or item.wall is None:
            continue
        if (item.wall, str(_as_mapping(overhead_item).get("run_role") or "")) in base_run_keys:
            continue
        violations.append(
            RuleViolation(
                rule_id="LAYOUT-05",
                severity="hard",
                text=f"{item.key} is a wall cabinet without base support below.",
            )
        )
    return tuple(violations)


def _validate_layout_06(variant: Mapping[str, object]) -> tuple[RuleViolation, ...]:
    violations: list[RuleViolation] = []
    for run in _runs_from_variant(variant):
        run_start = _as_float(run.get("start_mm"))
        run_end = _as_float(run.get("end_mm"))
        items = [
            _item_from_payload(_as_mapping(raw_item))
            for raw_item in _as_list(run.get("items"))
            if isinstance(raw_item, Mapping)
        ]
        for item in items:
            if item.component not in {"fridge", "tall_cabinet"}:
                continue
            if item.start_mm is None or item.end_mm is None:
                continue
            near_corner = (
                abs(item.start_mm - run_start) <= CORNER_PLACEMENT_TOLERANCE_MM
                or abs(run_end - item.end_mm) <= CORNER_PLACEMENT_TOLERANCE_MM
            )
            if near_corner:
                continue
            violations.append(
                RuleViolation(
                    rule_id="LAYOUT-06",
                    severity="warning",
                    text=(
                        f"{item.key} should sit near a layout corner; it is "
                        f"{item.start_mm - run_start:.0f} mm from run start and "
                        f"{run_end - item.end_mm:.0f} mm from run end."
                    ),
                )
            )
    return tuple(violations)


def validate_variant(
    environment: Environment,
    variant: Mapping[str, object],
) -> VariantValidationResult:
    variant_id = str(variant.get("id") or "variant")
    items = _items_from_variant(variant)
    bounds = RoomBounds.from_environment(environment)
    violations: list[RuleViolation] = []
    violations.extend(_validate_nkba_cl_01(items, bounds))
    violations.extend(_validate_nkba_cl_02(environment, items))
    violations.extend(_validate_workflow_01(items))
    violations.extend(_validate_workflow_02(items))
    violations.extend(_validate_workflow_03(items))
    violations.extend(_validate_layout_01(environment, items))
    violations.extend(_validate_layout_02(items))
    violations.extend(_validate_layout_03(variant))
    violations.extend(_validate_layout_04(variant))
    violations.extend(_validate_layout_05(variant))
    violations.extend(_validate_layout_06(variant))
    return VariantValidationResult(
        variant_id=variant_id,
        violations=tuple(violations),
    )


def validate_variants(
    environment: Environment,
    variants: list[dict[str, object]],
) -> tuple[VariantValidationResult, ...]:
    return tuple(validate_variant(environment, variant) for variant in variants)


def flatten_validation_results(
    results: tuple[VariantValidationResult, ...],
) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for result in results:
        for violation in result.violations:
            payload = violation.to_payload()
            payload["variant_id"] = result.variant_id
            flattened.append(payload)
    return flattened
