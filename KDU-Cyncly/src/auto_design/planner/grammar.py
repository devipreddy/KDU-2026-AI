from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from auto_design.schemas.environment import WallAnchor
from auto_design.schemas.intent import LayoutFamilyCode


RunRole = Literal["primary", "return", "left_leg", "bridge", "right_leg"]
ZoneType = Literal["cooling", "storage", "preparation", "cleaning", "cooking"]
StepSpec: TypeAlias = tuple[ZoneType, str, int, str]

FAMILY_LABELS: Mapping[LayoutFamilyCode, str] = {
    "I": "I-shaped",
    "L": "L-shaped",
    "U": "U-shaped",
}
WALLS_PER_FAMILY: Mapping[LayoutFamilyCode, int] = {"I": 1, "L": 2, "U": 3}

I_GRAMMAR_VARIANTS: tuple[tuple[StepSpec, ...], ...] = (
    (
        ("cooling", "fridge", 0, "Anchor cold storage at the run edge."),
        ("storage", "base_cabinets", 0, "Use base storage before the wet zone."),
        ("cleaning", "sink", 0, "Place cleaning in the middle of the wall run."),
        ("preparation", "prep_counter", 0, "Keep prep space between sink and cooking."),
        ("cooking", "stove", 0, "Finish with cooking at the opposite edge."),
    ),
    (
        ("storage", "base_cabinets", 0, "Begin with storage to terminate the run cleanly."),
        ("cooling", "fridge", 0, "Keep refrigeration near storage."),
        ("preparation", "prep_counter", 0, "Provide landing space before cleaning."),
        ("cleaning", "sink", 0, "Center cleaning inside the continuous run."),
        ("cooking", "stove", 0, "Place cooking after the preparation zone."),
    ),
    (
        ("cooking", "stove", 0, "Start with cooking at one edge of the linear run."),
        ("preparation", "prep_counter", 0, "Buffer cooking from cleaning."),
        ("cleaning", "sink", 0, "Keep cleaning central for dishwasher adjacency later."),
        ("storage", "base_cabinets", 0, "Use base storage as a run terminator."),
        ("cooling", "fridge", 0, "Finish with cooling at the opposite edge."),
    ),
)

L_GRAMMAR_VARIANTS: tuple[tuple[StepSpec, ...], ...] = (
    (
        ("cooling", "fridge", 0, "Place refrigerator near the outside corner."),
        ("storage", "base_cabinets", 0, "Continue the first leg with base storage."),
        ("cleaning", "sink", 0, "Keep cleaning on the primary leg."),
        ("preparation", "corner_prep", 1, "Use the corner turn as preparation flow."),
        ("cooking", "stove", 1, "Move cooking onto the return leg."),
    ),
    (
        ("storage", "base_cabinets", 0, "Terminate the primary leg with base storage."),
        ("cleaning", "sink", 0, "Keep sink and dishwasher adjacency on one leg."),
        ("preparation", "corner_prep", 1, "Reserve the turn for preparation."),
        ("cooking", "stove", 1, "Place cooking on the return leg."),
        ("cooling", "fridge", 1, "Use the return edge for cooling."),
    ),
    (
        ("cooking", "stove", 0, "Put cooking on the primary leg for a compact triangle."),
        ("preparation", "prep_counter", 0, "Buffer cooking before the corner."),
        ("cleaning", "sink", 1, "Move cleaning onto the return leg."),
        ("storage", "base_cabinets", 1, "Continue the return with base storage."),
        ("cooling", "fridge", 1, "Terminate the return with cooling."),
    ),
)

U_GRAMMAR_VARIANTS: tuple[tuple[StepSpec, ...], ...] = (
    (
        ("cooling", "fridge", 0, "Anchor cooling at one outside leg."),
        ("storage", "base_cabinets", 0, "Continue the first leg with storage."),
        ("cleaning", "sink", 1, "Use the bridge wall for the wet zone."),
        ("preparation", "prep_counter", 1, "Reserve central counter for prep."),
        ("cooking", "stove", 2, "Place cooking on the opposite leg."),
    ),
    (
        ("cooking", "stove", 0, "Anchor cooking at one outside leg."),
        ("preparation", "prep_counter", 0, "Keep prep adjacent to cooking."),
        ("cleaning", "sink", 1, "Put cleaning on the bridge wall."),
        ("storage", "base_cabinets", 2, "Use the opposite leg for storage."),
        ("cooling", "fridge", 2, "Terminate the U with cooling."),
    ),
    (
        ("storage", "base_cabinets", 0, "Start with storage at the first outside leg."),
        ("cooling", "fridge", 0, "Keep cooling near storage."),
        ("preparation", "prep_counter", 1, "Use the bridge as shared prep space."),
        ("cleaning", "sink", 1, "Center cleaning on the bridge wall."),
        ("cooking", "stove", 2, "Place cooking on the final leg."),
    ),
)

GRAMMAR_VARIANTS: Mapping[LayoutFamilyCode, tuple[tuple[StepSpec, ...], ...]] = {
    "I": I_GRAMMAR_VARIANTS,
    "L": L_GRAMMAR_VARIANTS,
    "U": U_GRAMMAR_VARIANTS,
}


@dataclass(frozen=True)
class RunSegmentTemplate:
    wall: WallAnchor
    role: RunRole
    start_mm: float
    end_mm: float

    @property
    def length_mm(self) -> float:
        return self.end_mm - self.start_mm

    def to_payload(self) -> dict[str, object]:
        return {
            "wall": self.wall,
            "role": self.role,
            "start_mm": self.start_mm,
            "end_mm": self.end_mm,
            "length_mm": self.length_mm,
        }


@dataclass(frozen=True)
class GrammarStep:
    order: int
    zone_type: ZoneType
    component: str
    wall: WallAnchor
    rationale: str

    def to_payload(self) -> dict[str, object]:
        return {
            "order": self.order,
            "zone_type": self.zone_type,
            "component": self.component,
            "wall": self.wall,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class KitchenTopologyTemplate:
    id: str
    family: LayoutFamilyCode
    name: str
    walls: tuple[WallAnchor, ...]
    runs: tuple[RunSegmentTemplate, ...]
    steps: tuple[GrammarStep, ...]
    rationale: tuple[str, ...]

    @property
    def family_label(self) -> str:
        return FAMILY_LABELS[self.family]

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family": self.family,
            "family_label": self.family_label,
            "name": self.name,
            "walls": list(self.walls),
            "runs": [run.to_payload() for run in self.runs],
            "steps": [step.to_payload() for step in self.steps],
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class WallRunCandidate:
    wall: WallAnchor
    length_mm: float
    available_segments_mm: tuple[tuple[float, float], ...]

    @property
    def longest_segment(self) -> tuple[float, float]:
        if not self.available_segments_mm:
            return (0.0, self.length_mm)
        return max(self.available_segments_mm, key=lambda segment: segment[1] - segment[0])


def _family_from_payload(feasibility: Mapping[str, object]) -> LayoutFamilyCode | None:
    family = feasibility.get("selected_family")
    if family in {"I", "L", "U"}:
        return cast(LayoutFamilyCode, family)
    return None


def _candidate_walls_from_payload(
    feasibility: Mapping[str, object],
    family: LayoutFamilyCode,
) -> tuple[WallAnchor, ...]:
    topology_fits = feasibility.get("topology_fits")
    if not isinstance(topology_fits, Mapping):
        return ()
    fit = topology_fits.get(family)
    if not isinstance(fit, Mapping):
        return ()
    walls = fit.get("candidate_walls")
    if not isinstance(walls, list):
        return ()
    return tuple(wall for wall in walls if wall in {"north", "south", "east", "west"})


def _wall_runs_from_payload(
    feasibility: Mapping[str, object],
) -> dict[WallAnchor, WallRunCandidate]:
    raw_runs = feasibility.get("allowed_wall_runs")
    if not isinstance(raw_runs, list):
        return {}

    runs: dict[WallAnchor, WallRunCandidate] = {}
    for raw_run in raw_runs:
        if not isinstance(raw_run, Mapping):
            continue
        wall = raw_run.get("anchor")
        if wall not in {"north", "south", "east", "west"}:
            continue
        raw_segments = raw_run.get("available_segments_mm")
        segments: list[tuple[float, float]] = []
        if isinstance(raw_segments, list):
            for segment in raw_segments:
                if not isinstance(segment, Mapping):
                    continue
                segments.append(
                    (
                        float(segment.get("start_mm", 0.0)),
                        float(segment.get("end_mm", 0.0)),
                    )
                )
        runs[cast(WallAnchor, wall)] = WallRunCandidate(
            wall=cast(WallAnchor, wall),
            length_mm=float(raw_run.get("length_mm", 0.0)),
            available_segments_mm=tuple(segments),
        )
    return runs


def _run_roles(family: LayoutFamilyCode) -> tuple[RunRole, ...]:
    if family == "I":
        return ("primary",)
    if family == "L":
        return ("primary", "return")
    return ("left_leg", "bridge", "right_leg")


def _runs_for_walls(
    family: LayoutFamilyCode,
    walls: tuple[WallAnchor, ...],
    wall_runs: Mapping[WallAnchor, WallRunCandidate],
) -> tuple[RunSegmentTemplate, ...]:
    runs: list[RunSegmentTemplate] = []
    for wall, role in zip(walls, _run_roles(family), strict=True):
        candidate = wall_runs[wall]
        start, end = candidate.longest_segment
        runs.append(
            RunSegmentTemplate(
                wall=wall,
                role=role,
                start_mm=start,
                end_mm=end,
            )
        )
    return tuple(runs)


def _steps_for_variant(
    walls: tuple[WallAnchor, ...],
    specs: tuple[StepSpec, ...],
) -> tuple[GrammarStep, ...]:
    steps: list[GrammarStep] = []
    for index, (zone_type, component, wall_index, rationale) in enumerate(specs, start=1):
        safe_wall_index = min(wall_index, len(walls) - 1)
        steps.append(
            GrammarStep(
                order=index,
                zone_type=zone_type,
                component=component,
                wall=walls[safe_wall_index],
                rationale=rationale,
            )
        )
    return tuple(steps)


def _fallback_walls(
    family: LayoutFamilyCode,
    wall_runs: Mapping[WallAnchor, WallRunCandidate],
) -> tuple[WallAnchor, ...]:
    required_wall_count = WALLS_PER_FAMILY[family]
    ranked = sorted(
        wall_runs.values(),
        key=lambda candidate: candidate.longest_segment[1] - candidate.longest_segment[0],
        reverse=True,
    )
    return tuple(candidate.wall for candidate in ranked[:required_wall_count])


def generate_topology_templates(
    feasibility: Mapping[str, object],
    *,
    max_variants: int = 5,
) -> tuple[KitchenTopologyTemplate, ...]:
    family = _family_from_payload(feasibility)
    if family is None or max_variants <= 0:
        return ()

    wall_runs = _wall_runs_from_payload(feasibility)
    walls = _candidate_walls_from_payload(feasibility, family)
    if not walls:
        walls = _fallback_walls(family, wall_runs)
    required_wall_count = WALLS_PER_FAMILY[family]
    if len(walls) < required_wall_count:
        return ()
    walls = walls[:required_wall_count]

    if any(wall not in wall_runs for wall in walls):
        return ()

    runs = _runs_for_walls(family, walls, wall_runs)
    grammar_variants = GRAMMAR_VARIANTS[family][:max_variants]
    templates: list[KitchenTopologyTemplate] = []
    for index, specs in enumerate(grammar_variants, start=1):
        template_id = f"template-{family.lower()}-{index}"
        templates.append(
            KitchenTopologyTemplate(
                id=template_id,
                family=family,
                name=f"{FAMILY_LABELS[family]} procedural grammar {index}",
                walls=walls,
                runs=runs,
                steps=_steps_for_variant(walls, specs),
                rationale=(
                    f"Generated from the {FAMILY_LABELS[family]} topology grammar.",
                    "Uses only cabinet-enabled wall runs selected by feasibility.",
                    "Coordinates and SKU placements are deferred to deterministic geometry.",
                ),
            )
        )
    return tuple(templates)
