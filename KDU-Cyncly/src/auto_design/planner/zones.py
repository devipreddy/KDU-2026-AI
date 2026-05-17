from __future__ import annotations

from dataclasses import dataclass

from auto_design.planner.grammar import KitchenTopologyTemplate, RunRole, ZoneType
from auto_design.schemas.environment import WallAnchor
from auto_design.schemas.intent import LayoutFamilyCode, StructuredIntent


ZONE_ORDER: tuple[ZoneType, ...] = (
    "cooling",
    "storage",
    "cleaning",
    "preparation",
    "cooking",
)


@dataclass(frozen=True)
class ZoneAssignment:
    zone_type: ZoneType
    wall: WallAnchor
    run_role: RunRole
    components: tuple[str, ...]
    order: int
    rationale: str

    def to_payload(self) -> dict[str, object]:
        return {
            "zone_type": self.zone_type,
            "wall": self.wall,
            "run_role": self.run_role,
            "components": list(self.components),
            "order": self.order,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ItemZoneAssignment:
    item: str
    zone_type: ZoneType
    wall: WallAnchor
    run_role: RunRole
    required: bool
    reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "item": self.item,
            "zone_type": self.zone_type,
            "wall": self.wall,
            "run_role": self.run_role,
            "required": self.required,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MacroZonePlan:
    template_id: str
    family: LayoutFamilyCode
    zones: tuple[ZoneAssignment, ...]
    item_assignments: tuple[ItemZoneAssignment, ...]
    rationale: tuple[str, ...]

    def owner_for(self, item: str) -> ItemZoneAssignment | None:
        return next(
            (
                assignment
                for assignment in self.item_assignments
                if assignment.item == item
            ),
            None,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "template_id": self.template_id,
            "family": self.family,
            "zones": [zone.to_payload() for zone in self.zones],
            "item_assignments": [
                assignment.to_payload() for assignment in self.item_assignments
            ],
            "rationale": list(self.rationale),
        }


def _normalized_items(items: list[str]) -> set[str]:
    return {item.casefold() for item in items}


def _run_role_for(template: KitchenTopologyTemplate, wall: WallAnchor) -> RunRole:
    for run in template.runs:
        if run.wall == wall:
            return run.role
    return template.runs[0].role


def _primary_step_for_zone(template: KitchenTopologyTemplate) -> dict[ZoneType, int]:
    zone_indexes: dict[ZoneType, int] = {}
    for index, step in enumerate(template.steps):
        zone_indexes.setdefault(step.zone_type, index)
    return zone_indexes


def _sink_component(required: set[str], excluded: set[str]) -> str:
    if "double_sink" in required and "double_sink" not in excluded:
        return "double_sink"
    if "single_sink" in required or "double_sink" in excluded:
        return "single_sink"
    return "sink"


def _components_for_zone(
    zone_type: ZoneType,
    intent: StructuredIntent,
    *,
    required: set[str],
    excluded: set[str],
) -> tuple[str, ...]:
    if zone_type == "cooling":
        return () if "fridge" in excluded else ("fridge",)
    if zone_type == "cleaning":
        components = [_sink_component(required, excluded)]
        if "dishwasher" not in excluded:
            components.append("dishwasher")
        return tuple(components)
    if zone_type == "cooking":
        components: list[str] = []
        if "stove" not in excluded:
            components.append("stove")
        if "hood" not in excluded:
            components.append("hood")
        return tuple(components)
    if zone_type == "preparation":
        return ("prep_counter", "prep_base_cabinet")

    components = ["base_cabinet"]
    if intent.base_cabinets_only:
        return tuple(components)
    if intent.upper_cabinets:
        components.append("wall_cabinet")
    if intent.pantry_storage or intent.tall_cabinets:
        components.append("tall_cabinet")
    return tuple(components)


def _is_required(component: str, required: set[str]) -> bool:
    core_components = {
        "sink",
        "single_sink",
        "double_sink",
        "stove",
        "fridge",
        "prep_counter",
        "prep_base_cabinet",
        "base_cabinet",
    }
    if component in core_components:
        return True
    if component == "wall_cabinet":
        return "upper_cabinets" in required or "wall_cabinet" in required
    if component == "tall_cabinet":
        return "pantry" in required or "tall_cabinet" in required
    return component in required


def plan_zones_for_template(
    template: KitchenTopologyTemplate,
    intent: StructuredIntent,
) -> MacroZonePlan:
    required = _normalized_items(intent.required_items)
    excluded = _normalized_items(intent.excluded_items)
    primary_steps = _primary_step_for_zone(template)
    zones: list[ZoneAssignment] = []
    item_assignments: list[ItemZoneAssignment] = []

    for order, zone_type in enumerate(ZONE_ORDER, start=1):
        step_index = primary_steps[zone_type]
        step = template.steps[step_index]
        run_role = _run_role_for(template, step.wall)
        components = _components_for_zone(
            zone_type,
            intent,
            required=required,
            excluded=excluded,
        )
        zones.append(
            ZoneAssignment(
                zone_type=zone_type,
                wall=step.wall,
                run_role=run_role,
                components=components,
                order=order,
                rationale=step.rationale,
            )
        )
        for component in components:
            item_assignments.append(
                ItemZoneAssignment(
                    item=component,
                    zone_type=zone_type,
                    wall=step.wall,
                    run_role=run_role,
                    required=_is_required(component, required),
                    reason=f"{component} belongs to the {zone_type} zone before placement.",
                )
            )

    return MacroZonePlan(
        template_id=template.id,
        family=template.family,
        zones=tuple(zones),
        item_assignments=tuple(item_assignments),
        rationale=(
            "Macro zones are assigned after topology selection and before item placement.",
            "Sink and dishwasher share cleaning ownership.",
            "Stove and hood share cooking ownership.",
            "Fridge owns cooling; prep and storage are assigned to cabinet runs.",
        ),
    )
