from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from auto_design.catalog.service import CatalogService
from auto_design.planner.grammar import KitchenTopologyTemplate, generate_topology_templates
from auto_design.planner.placement import PlacementPlan, generate_placement_plan
from auto_design.planner.zones import MacroZonePlan, plan_zones_for_template
from auto_design.schemas.input import DesignInput
from auto_design.schemas.intent import StructuredIntent


MIN_VARIANT_COUNT = 3
MAX_VARIANT_COUNT = 5


@dataclass(frozen=True)
class GeneratedVariant:
    index: int
    seed: int
    template: KitchenTopologyTemplate
    zone_plan: MacroZonePlan
    placement_plan: PlacementPlan

    def to_payload(self) -> dict[str, object]:
        return {
            "id": f"variant-{self.template.family.lower()}-{self.index}",
            "family": self.template.family,
            "family_label": self.template.family_label,
            "status": "placed_template",
            "template_id": self.template.id,
            "topology": self.template.to_payload(),
            "zone_plan": self.zone_plan.to_payload(),
            "placement": self.placement_plan.to_payload(),
            "layout": self.placement_plan.layout_payload(),
            "variant_seed": self.seed,
            "diversity": _diversity_payload(self.zone_plan, self.placement_plan),
            "notes": "Seeded async worker generated a deterministic variant.",
        }


def variant_seed_for(
    design_input: DesignInput,
    intent: StructuredIntent,
    feasibility: dict[str, object],
) -> int:
    payload = {
        "environment": design_input.environment.model_dump(mode="json"),
        "prompt": design_input.preferences.prompt,
        "intent": intent.model_dump(mode="json"),
        "selected_family": feasibility.get("selected_family"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16)


def _bounded_variant_count(count: int) -> int:
    return max(MIN_VARIANT_COUNT, min(MAX_VARIANT_COUNT, count))


def _component_wall_map(zone_plan: MacroZonePlan) -> dict[str, str]:
    tracked = {
        "dishwasher",
        "double_sink",
        "fridge",
        "hood",
        "single_sink",
        "sink",
        "stove",
        "tall_cabinet",
    }
    return {
        assignment.item: assignment.wall
        for assignment in zone_plan.item_assignments
        if assignment.item in tracked
    }


def _run_sequences(placement_plan: PlacementPlan) -> dict[str, list[str]]:
    return {
        f"{run.wall}:{run.run_role}": [
            item.component
            for item in run.items
        ]
        for run in placement_plan.runs
    }


def _component_offsets(placement_plan: PlacementPlan) -> dict[str, dict[str, object]]:
    tracked = {"dishwasher", "fridge", "sink", "single_sink", "double_sink", "stove"}
    offsets: dict[str, dict[str, object]] = {}
    for item in placement_plan.items:
        if item.component not in tracked:
            continue
        offsets[item.component] = {
            "wall": item.wall,
            "start_mm": item.start_mm,
            "end_mm": item.end_mm,
        }
    return offsets


def _diversity_signature(
    zone_plan: MacroZonePlan,
    placement_plan: PlacementPlan,
) -> str:
    wall_map = _component_wall_map(zone_plan)
    sequences = _run_sequences(placement_plan)
    offsets = _component_offsets(placement_plan)
    payload: dict[str, Any] = {
        "walls": wall_map,
        "sequences": sequences,
        "offsets": offsets,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _diversity_payload(
    zone_plan: MacroZonePlan,
    placement_plan: PlacementPlan,
) -> dict[str, object]:
    return {
        "signature": _diversity_signature(zone_plan, placement_plan),
        "zone_wall_map": _component_wall_map(zone_plan),
        "run_sequences": _run_sequences(placement_plan),
        "component_offsets": _component_offsets(placement_plan),
    }


async def _generate_variant_worker(
    *,
    index: int,
    seed: int,
    template: KitchenTopologyTemplate,
    design_input: DesignInput,
    intent: StructuredIntent,
    catalog: CatalogService,
) -> GeneratedVariant:
    zone_plan = plan_zones_for_template(template, intent)
    placement_plan = await asyncio.to_thread(
        generate_placement_plan,
        design_input.environment,
        template,
        zone_plan,
        catalog,
        intent,
    )
    return GeneratedVariant(
        index=index,
        seed=seed,
        template=template,
        zone_plan=zone_plan,
        placement_plan=placement_plan,
    )


async def generate_layout_variants_async(
    *,
    design_input: DesignInput,
    intent: StructuredIntent,
    feasibility: dict[str, object],
    catalog: CatalogService,
    count: int = MAX_VARIANT_COUNT,
) -> list[dict[str, object]]:
    requested_count = _bounded_variant_count(count)
    seed = variant_seed_for(design_input, intent, feasibility)
    templates = generate_topology_templates(
        feasibility,
        max_variants=requested_count,
        seed=seed,
    )
    if not templates:
        family = feasibility.get("selected_family") or intent.layout_family
        if family is None:
            return []
        return [
            {
                "id": "variant-template-unavailable",
                "family": family,
                "status": "template_unavailable",
                "variant_seed": seed,
                "notes": "No procedural topology template could be derived from feasibility.",
            }
        ]

    workers = [
        _generate_variant_worker(
            index=index,
            seed=seed + index,
            template=template,
            design_input=design_input,
            intent=intent,
            catalog=catalog,
        )
        for index, template in enumerate(templates, start=1)
    ]
    generated = await asyncio.gather(*workers)
    return [
        variant.to_payload()
        for variant in sorted(generated, key=lambda item: item.index)
    ]
