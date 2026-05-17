from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from auto_design.catalog.service import CatalogService
from auto_design.geometry import RoomBounds, build_opening_constraints
from auto_design.schemas.catalog import Product
from auto_design.schemas.environment import Environment, WallAnchor
from auto_design.schemas.input import DesignInput
from auto_design.schemas.intent import LayoutFamilyCode, StructuredIntent


STANDARD_BASE_DEPTH_MM = 600.0
MIN_WALKWAY_CLEARANCE_MM = 1067.0
MIN_PREP_COUNTER_MM = 600.0
MIN_I_RUN_MM = 1800.0
MIN_TOPOLOGY_LEG_MM = 1200.0

FeasibilityStatus = Literal["feasible", "fallback", "infeasible"]

ADJACENT_WALL_PAIRS: tuple[tuple[WallAnchor, WallAnchor], ...] = (
    ("north", "east"),
    ("east", "south"),
    ("south", "west"),
    ("west", "north"),
)
U_WALL_SETS: tuple[tuple[WallAnchor, WallAnchor, WallAnchor], ...] = (
    ("north", "east", "south"),
    ("east", "south", "west"),
    ("south", "west", "north"),
    ("west", "north", "east"),
)
DEFAULT_FAMILY_PREFERENCE: tuple[LayoutFamilyCode, ...] = ("L", "I", "U")


@dataclass(frozen=True)
class AvailableWallRun:
    anchor: WallAnchor
    name: str
    length_mm: float
    available_segments_mm: tuple[tuple[float, float], ...]

    @property
    def available_run_mm(self) -> float:
        return sum(end - start for start, end in self.available_segments_mm)

    @property
    def max_contiguous_run_mm(self) -> float:
        if not self.available_segments_mm:
            return 0.0
        return max(end - start for start, end in self.available_segments_mm)

    def to_payload(self) -> dict[str, object]:
        return {
            "anchor": self.anchor,
            "name": self.name,
            "length_mm": self.length_mm,
            "available_run_mm": self.available_run_mm,
            "max_contiguous_run_mm": self.max_contiguous_run_mm,
            "available_segments_mm": [
                {"start_mm": start, "end_mm": end}
                for start, end in self.available_segments_mm
            ],
        }


@dataclass(frozen=True)
class RequiredFootprintItem:
    kind: str
    product_id: str | None
    product_type: str | None
    width_mm: float
    counted_in_run: bool
    reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "product_id": self.product_id,
            "product_type": self.product_type,
            "width_mm": self.width_mm,
            "counted_in_run": self.counted_in_run,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TopologyFit:
    family: LayoutFamilyCode
    feasible: bool
    candidate_walls: tuple[WallAnchor, ...]
    available_run_mm: float
    max_contiguous_run_mm: float
    walkway_clearance_mm: float
    required_run_mm: float
    minimum_wall_length_mm: float
    reasons: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "feasible": self.feasible,
            "candidate_walls": list(self.candidate_walls),
            "available_run_mm": self.available_run_mm,
            "max_contiguous_run_mm": self.max_contiguous_run_mm,
            "walkway_clearance_mm": self.walkway_clearance_mm,
            "required_run_mm": self.required_run_mm,
            "minimum_wall_length_mm": self.minimum_wall_length_mm,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FeasibilityResult:
    status: FeasibilityStatus
    feasible: bool
    requested_family: LayoutFamilyCode | None
    selected_family: LayoutFamilyCode | None
    fallback_family: LayoutFamilyCode | None
    fallback_reason: str | None
    required_appliance_footprint_mm: float
    required_run_mm: float
    required_items: tuple[RequiredFootprintItem, ...]
    allowed_wall_runs: tuple[AvailableWallRun, ...]
    topology_fits: dict[LayoutFamilyCode, TopologyFit]
    notes: tuple[str, ...]

    @property
    def requested_family_feasible(self) -> bool | None:
        if self.requested_family is None:
            return None
        return self.topology_fits[self.requested_family].feasible

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "feasible": self.feasible,
            "requested_family": self.requested_family,
            "requested_family_feasible": self.requested_family_feasible,
            "selected_family": self.selected_family,
            "fallback_family": self.fallback_family,
            "fallback_reason": self.fallback_reason,
            "required_appliance_footprint_mm": self.required_appliance_footprint_mm,
            "required_run_mm": self.required_run_mm,
            "required_items": [item.to_payload() for item in self.required_items],
            "allowed_cabinet_walls": [wall.name for wall in self.allowed_wall_runs],
            "allowed_wall_anchors": [wall.anchor for wall in self.allowed_wall_runs],
            "allowed_wall_runs": [wall.to_payload() for wall in self.allowed_wall_runs],
            "topology_fits": {
                family: fit.to_payload() for family, fit in self.topology_fits.items()
            },
            "constraints": {
                "standard_base_depth_mm": STANDARD_BASE_DEPTH_MM,
                "minimum_walkway_clearance_mm": MIN_WALKWAY_CLEARANCE_MM,
                "minimum_prep_counter_mm": MIN_PREP_COUNTER_MM,
                "minimum_i_run_mm": MIN_I_RUN_MM,
                "minimum_topology_leg_mm": MIN_TOPOLOGY_LEG_MM,
            },
            "notes": list(self.notes),
        }


def _clip_span(start: float, end: float, length: float) -> tuple[float, float] | None:
    clipped_start = min(max(start, 0.0), length)
    clipped_end = min(max(end, 0.0), length)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def _available_segments(
    length_mm: float,
    blocked_spans: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    clipped = sorted(
        span
        for span in (
            _clip_span(start, end, length_mm) for start, end in blocked_spans
        )
        if span is not None
    )
    segments: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in clipped:
        if start > cursor:
            segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length_mm:
        segments.append((cursor, length_mm))
    return tuple(segments)


def allowed_wall_runs(environment: Environment) -> tuple[AvailableWallRun, ...]:
    bounds = RoomBounds.from_environment(environment)
    opening_constraints = build_opening_constraints(environment)
    runs: list[AvailableWallRun] = []

    for wall in environment.wall:
        if wall.has_cabinets is False or wall.anchor is None:
            continue
        wall_system = bounds.wall(wall.anchor)
        length_mm = (
            float(wall.dimensions.length_mm)
            if wall.dimensions.length_mm is not None
            else wall_system.length_mm
        )
        blocked_spans = tuple(
            (span.start_mm, span.end_mm)
            for span in opening_constraints.blocked_spans_for_wall(wall.anchor)
        )
        runs.append(
            AvailableWallRun(
                anchor=wall.anchor,
                name=wall.name,
                length_mm=length_mm,
                available_segments_mm=_available_segments(length_mm, blocked_spans),
            )
        )
    return tuple(runs)


def _products_matching(catalog: CatalogService, *needles: str) -> tuple[Product, ...]:
    lowered_needles = tuple(needle.casefold() for needle in needles)
    return tuple(
        product
        for product in catalog.products
        if any(needle in product.type.casefold() for needle in lowered_needles)
    )


def _narrowest_product(catalog: CatalogService, *needles: str) -> Product | None:
    candidates = _products_matching(catalog, *needles)
    if not candidates:
        return None
    return min(candidates, key=lambda product: float(product.width_mm))


def _append_footprint_item(
    items: list[RequiredFootprintItem],
    *,
    catalog: CatalogService,
    kind: str,
    needles: Sequence[str],
    counted_in_run: bool = True,
    fallback_width_mm: float = 600.0,
    reason: str,
) -> None:
    product = _narrowest_product(catalog, *needles)
    width_mm = float(product.width_mm) if product is not None and counted_in_run else 0.0
    if product is None and counted_in_run:
        width_mm = fallback_width_mm
    items.append(
        RequiredFootprintItem(
            kind=kind,
            product_id=product.id if product is not None else None,
            product_type=product.type if product is not None else None,
            width_mm=width_mm,
            counted_in_run=counted_in_run,
            reason=reason if product is not None else f"{reason}; catalog candidate missing",
        )
    )


def required_footprint_items(
    catalog: CatalogService,
    intent: StructuredIntent,
) -> tuple[RequiredFootprintItem, ...]:
    required = {item.casefold() for item in intent.required_items}
    excluded = {item.casefold() for item in intent.excluded_items}
    items: list[RequiredFootprintItem] = []

    sink_needles = ("sink_single", "sink")
    sink_kind = "sink"
    if "double_sink" in required and "double_sink" not in excluded:
        sink_needles = ("sink_double",)
        sink_kind = "double_sink"
    elif "single_sink" in required or "double_sink" in excluded:
        sink_needles = ("sink_single",)
        sink_kind = "single_sink"

    _append_footprint_item(
        items,
        catalog=catalog,
        kind=sink_kind,
        needles=sink_needles,
        reason="Core cleaning zone requires a sink footprint.",
    )
    _append_footprint_item(
        items,
        catalog=catalog,
        kind="stove",
        needles=("stove", "cooktop", "range"),
        reason="Core cooking zone requires a stove footprint.",
    )
    _append_footprint_item(
        items,
        catalog=catalog,
        kind="fridge",
        needles=("fridge", "refrigerator"),
        fallback_width_mm=700.0,
        reason="Core cooling zone requires a refrigerator footprint.",
    )

    if "dishwasher" in required and "dishwasher" not in excluded:
        _append_footprint_item(
            items,
            catalog=catalog,
            kind="dishwasher",
            needles=("dishwasher",),
            reason="Prompt or preferences require a dishwasher.",
        )
    if "oven" in required and "oven" not in excluded:
        _append_footprint_item(
            items,
            catalog=catalog,
            kind="oven",
            needles=("oven",),
            reason="Prompt or preferences require an oven.",
        )
    if "microwave" in required and "microwave" not in excluded:
        _append_footprint_item(
            items,
            catalog=catalog,
            kind="microwave",
            needles=("microwave",),
            reason="Prompt or preferences require a microwave.",
        )
    if "hood" in required and "hood" not in excluded:
        _append_footprint_item(
            items,
            catalog=catalog,
            kind="hood",
            needles=("hood",),
            counted_in_run=False,
            reason="Hood is validated vertically above the cooking zone, not as base run.",
        )
    if intent.pantry_storage or intent.tall_cabinets:
        _append_footprint_item(
            items,
            catalog=catalog,
            kind="tall_cabinet",
            needles=("tall_cabinet",),
            reason="Prompt requires pantry or tall cabinet storage.",
        )
    return tuple(items)


def required_appliance_footprint_mm(items: Iterable[RequiredFootprintItem]) -> float:
    return sum(item.width_mm for item in items if item.counted_in_run)


def _runs_by_anchor(runs: Iterable[AvailableWallRun]) -> dict[WallAnchor, AvailableWallRun]:
    return {run.anchor: run for run in runs}


def _walkway_for_single_wall(anchor: WallAnchor, bounds: RoomBounds) -> float:
    if anchor in {"north", "south"}:
        return bounds.depth - STANDARD_BASE_DEPTH_MM
    return bounds.width - STANDARD_BASE_DEPTH_MM


def _walkway_for_l_pair(pair: tuple[WallAnchor, WallAnchor], bounds: RoomBounds) -> float:
    clearances = [_walkway_for_single_wall(anchor, bounds) for anchor in pair]
    return min(clearances)


def _walkway_for_u_set(
    walls: tuple[WallAnchor, WallAnchor, WallAnchor],
    bounds: RoomBounds,
) -> float:
    clearances: list[float] = []
    wall_set = set(walls)
    if {"north", "south"}.issubset(wall_set):
        clearances.append(bounds.depth - (STANDARD_BASE_DEPTH_MM * 2.0))
    if {"east", "west"}.issubset(wall_set):
        clearances.append(bounds.width - (STANDARD_BASE_DEPTH_MM * 2.0))
    return min(clearances) if clearances else -STANDARD_BASE_DEPTH_MM


def _fit_i(
    runs: tuple[AvailableWallRun, ...],
    bounds: RoomBounds,
    required_run_mm: float,
) -> TopologyFit:
    reasons: list[str] = []
    minimum_run_mm = max(required_run_mm, MIN_I_RUN_MM)
    viable = [
        run
        for run in runs
        if run.max_contiguous_run_mm >= minimum_run_mm
        and _walkway_for_single_wall(run.anchor, bounds) >= MIN_WALKWAY_CLEARANCE_MM
    ]

    if viable:
        best = max(viable, key=lambda run: run.max_contiguous_run_mm)
        walkway = _walkway_for_single_wall(best.anchor, bounds)
        return TopologyFit(
            family="I",
            feasible=True,
            candidate_walls=(best.anchor,),
            available_run_mm=best.max_contiguous_run_mm,
            max_contiguous_run_mm=best.max_contiguous_run_mm,
            walkway_clearance_mm=walkway,
            required_run_mm=required_run_mm,
            minimum_wall_length_mm=minimum_run_mm,
            reasons=(),
        )

    best_run = max((run.max_contiguous_run_mm for run in runs), default=0.0)
    best_walkway = max(
        (_walkway_for_single_wall(run.anchor, bounds) for run in runs),
        default=0.0,
    )
    if not runs:
        reasons.append("requires at least one cabinet-enabled anchored wall")
    if best_run < minimum_run_mm:
        reasons.append(
            f"requires one continuous run of at least {minimum_run_mm:.0f} mm"
        )
    if best_walkway < MIN_WALKWAY_CLEARANCE_MM:
        reasons.append(
            f"requires at least {MIN_WALKWAY_CLEARANCE_MM:.0f} mm walkway clearance"
        )
    return TopologyFit(
        family="I",
        feasible=False,
        candidate_walls=(),
        available_run_mm=best_run,
        max_contiguous_run_mm=best_run,
        walkway_clearance_mm=best_walkway,
        required_run_mm=required_run_mm,
        minimum_wall_length_mm=minimum_run_mm,
        reasons=tuple(reasons),
    )


def _fit_l(
    runs: tuple[AvailableWallRun, ...],
    bounds: RoomBounds,
    required_run_mm: float,
) -> TopologyFit:
    by_anchor = _runs_by_anchor(runs)
    candidates: list[tuple[tuple[WallAnchor, WallAnchor], float, float, float]] = []
    for pair in ADJACENT_WALL_PAIRS:
        if not all(anchor in by_anchor for anchor in pair):
            continue
        pair_runs = (by_anchor[pair[0]], by_anchor[pair[1]])
        if any(run.max_contiguous_run_mm < MIN_TOPOLOGY_LEG_MM for run in pair_runs):
            continue
        available = sum(run.available_run_mm for run in pair_runs)
        contiguous = min(run.max_contiguous_run_mm for run in pair_runs)
        walkway = _walkway_for_l_pair(pair, bounds)
        if available >= required_run_mm and walkway >= MIN_WALKWAY_CLEARANCE_MM:
            candidates.append((pair, available, contiguous, walkway))

    if candidates:
        pair, available, contiguous, walkway = max(candidates, key=lambda item: item[1])
        return TopologyFit(
            family="L",
            feasible=True,
            candidate_walls=pair,
            available_run_mm=available,
            max_contiguous_run_mm=contiguous,
            walkway_clearance_mm=walkway,
            required_run_mm=required_run_mm,
            minimum_wall_length_mm=MIN_TOPOLOGY_LEG_MM,
            reasons=(),
        )

    available_pairs = [
        pair for pair in ADJACENT_WALL_PAIRS if all(anchor in by_anchor for anchor in pair)
    ]
    best_available = max(
        (sum(by_anchor[anchor].available_run_mm for anchor in pair) for pair in available_pairs),
        default=0.0,
    )
    best_contiguous = max(
        (
            min(by_anchor[anchor].max_contiguous_run_mm for anchor in pair)
            for pair in available_pairs
        ),
        default=0.0,
    )
    best_walkway = max(
        (_walkway_for_l_pair(pair, bounds) for pair in available_pairs),
        default=0.0,
    )
    reasons: list[str] = []
    if not available_pairs:
        reasons.append("requires two adjacent cabinet-enabled anchored walls")
    if best_contiguous < MIN_TOPOLOGY_LEG_MM:
        reasons.append(
            f"requires each L leg to have at least {MIN_TOPOLOGY_LEG_MM:.0f} mm usable run"
        )
    if best_available < required_run_mm:
        reasons.append(f"requires {required_run_mm:.0f} mm total run for required items")
    if best_walkway < MIN_WALKWAY_CLEARANCE_MM:
        reasons.append(
            f"requires at least {MIN_WALKWAY_CLEARANCE_MM:.0f} mm walkway clearance"
        )
    return TopologyFit(
        family="L",
        feasible=False,
        candidate_walls=(),
        available_run_mm=best_available,
        max_contiguous_run_mm=best_contiguous,
        walkway_clearance_mm=best_walkway,
        required_run_mm=required_run_mm,
        minimum_wall_length_mm=MIN_TOPOLOGY_LEG_MM,
        reasons=tuple(reasons),
    )


def _fit_u(
    runs: tuple[AvailableWallRun, ...],
    bounds: RoomBounds,
    required_run_mm: float,
) -> TopologyFit:
    by_anchor = _runs_by_anchor(runs)
    candidates: list[tuple[tuple[WallAnchor, WallAnchor, WallAnchor], float, float, float]] = []
    for walls in U_WALL_SETS:
        if not all(anchor in by_anchor for anchor in walls):
            continue
        wall_runs = tuple(by_anchor[anchor] for anchor in walls)
        if any(run.max_contiguous_run_mm < MIN_TOPOLOGY_LEG_MM for run in wall_runs):
            continue
        available = sum(run.available_run_mm for run in wall_runs)
        contiguous = min(run.max_contiguous_run_mm for run in wall_runs)
        walkway = _walkway_for_u_set(walls, bounds)
        if available >= required_run_mm and walkway >= MIN_WALKWAY_CLEARANCE_MM:
            candidates.append((walls, available, contiguous, walkway))

    if candidates:
        walls, available, contiguous, walkway = max(candidates, key=lambda item: item[1])
        return TopologyFit(
            family="U",
            feasible=True,
            candidate_walls=walls,
            available_run_mm=available,
            max_contiguous_run_mm=contiguous,
            walkway_clearance_mm=walkway,
            required_run_mm=required_run_mm,
            minimum_wall_length_mm=MIN_TOPOLOGY_LEG_MM,
            reasons=(),
        )

    available_sets = [
        walls for walls in U_WALL_SETS if all(anchor in by_anchor for anchor in walls)
    ]
    best_available = max(
        (sum(by_anchor[anchor].available_run_mm for anchor in walls) for walls in available_sets),
        default=0.0,
    )
    best_contiguous = max(
        (
            min(by_anchor[anchor].max_contiguous_run_mm for anchor in walls)
            for walls in available_sets
        ),
        default=0.0,
    )
    best_walkway = max(
        (_walkway_for_u_set(walls, bounds) for walls in available_sets),
        default=0.0,
    )
    reasons: list[str] = []
    if not available_sets:
        reasons.append("requires three cabinet-enabled anchored walls")
    if best_contiguous < MIN_TOPOLOGY_LEG_MM:
        reasons.append(
            f"requires each U leg to have at least {MIN_TOPOLOGY_LEG_MM:.0f} mm usable run"
        )
    if best_available < required_run_mm:
        reasons.append(f"requires {required_run_mm:.0f} mm total run for required items")
    if best_walkway < MIN_WALKWAY_CLEARANCE_MM:
        reasons.append(
            f"requires at least {MIN_WALKWAY_CLEARANCE_MM:.0f} mm between opposing runs"
        )
    return TopologyFit(
        family="U",
        feasible=False,
        candidate_walls=(),
        available_run_mm=best_available,
        max_contiguous_run_mm=best_contiguous,
        walkway_clearance_mm=best_walkway,
        required_run_mm=required_run_mm,
        minimum_wall_length_mm=MIN_TOPOLOGY_LEG_MM,
        reasons=tuple(reasons),
    )


def analyze_topology_fits(
    environment: Environment,
    runs: tuple[AvailableWallRun, ...],
    required_run_mm: float,
) -> dict[LayoutFamilyCode, TopologyFit]:
    bounds = RoomBounds.from_environment(environment)
    return {
        "I": _fit_i(runs, bounds, required_run_mm),
        "L": _fit_l(runs, bounds, required_run_mm),
        "U": _fit_u(runs, bounds, required_run_mm),
    }


def _select_family(
    fits: dict[LayoutFamilyCode, TopologyFit],
    requested: LayoutFamilyCode | None,
) -> tuple[FeasibilityStatus, LayoutFamilyCode | None, LayoutFamilyCode | None, str | None]:
    if requested is not None and fits[requested].feasible:
        return "feasible", requested, None, None

    for family in DEFAULT_FAMILY_PREFERENCE:
        if fits[family].feasible:
            if requested is None:
                return "feasible", family, None, None
            reasons = "; ".join(fits[requested].reasons) or "requested family is not viable"
            return "fallback", family, family, f"Requested {requested} layout rejected: {reasons}."

    if requested is None:
        return "infeasible", None, None, "No supported topology can satisfy the room constraints."
    reasons = "; ".join(fits[requested].reasons) or "requested family is not viable"
    return "infeasible", None, None, f"Requested {requested} layout rejected: {reasons}."


def analyze_feasibility(
    design_input: DesignInput,
    intent: StructuredIntent,
    catalog: CatalogService,
) -> FeasibilityResult:
    runs = allowed_wall_runs(design_input.environment)
    footprint_items = required_footprint_items(catalog, intent)
    appliance_footprint = required_appliance_footprint_mm(footprint_items)
    required_run_mm = appliance_footprint + MIN_PREP_COUNTER_MM
    fits = analyze_topology_fits(design_input.environment, runs, required_run_mm)
    status, selected_family, fallback_family, fallback_reason = _select_family(
        fits,
        intent.layout_family,
    )

    notes: list[str] = [
        "Feasibility is checked before variant generation so impossible topology "
        "requests can degrade gracefully.",
        f"Required appliance footprint is {appliance_footprint:.0f} mm plus "
        f"{MIN_PREP_COUNTER_MM:.0f} mm preparation run.",
    ]
    if fallback_reason:
        notes.append(fallback_reason)
    if not runs:
        notes.append("No cabinet-enabled anchored walls are available for planning.")

    return FeasibilityResult(
        status=status,
        feasible=selected_family is not None,
        requested_family=intent.layout_family,
        selected_family=selected_family,
        fallback_family=fallback_family,
        fallback_reason=fallback_reason,
        required_appliance_footprint_mm=appliance_footprint,
        required_run_mm=required_run_mm,
        required_items=footprint_items,
        allowed_wall_runs=runs,
        topology_fits=fits,
        notes=tuple(notes),
    )
