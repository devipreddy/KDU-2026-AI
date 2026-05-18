from __future__ import annotations

from auto_design.planner import plan_zones_for_template
from auto_design.planner.grammar import GrammarStep, KitchenTopologyTemplate, RunSegmentTemplate
from auto_design.schemas import StructuredIntent


def l_shape_template() -> KitchenTopologyTemplate:
    return KitchenTopologyTemplate(
        id="template-l-test",
        family="L",
        name="L-shaped test grammar",
        walls=("north", "east"),
        runs=(
            RunSegmentTemplate(wall="north", role="primary", start_mm=0, end_mm=3600),
            RunSegmentTemplate(wall="east", role="return", start_mm=0, end_mm=3200),
        ),
        steps=(
            GrammarStep(
                order=1,
                zone_type="cooling",
                component="fridge",
                wall="north",
                rationale="Cooling starts the primary leg.",
            ),
            GrammarStep(
                order=2,
                zone_type="storage",
                component="base_cabinets",
                wall="north",
                rationale="Storage continues the primary leg.",
            ),
            GrammarStep(
                order=3,
                zone_type="cleaning",
                component="sink",
                wall="north",
                rationale="Cleaning stays on the primary leg.",
            ),
            GrammarStep(
                order=4,
                zone_type="preparation",
                component="corner_prep",
                wall="east",
                rationale="Preparation turns the corner.",
            ),
            GrammarStep(
                order=5,
                zone_type="cooking",
                component="stove",
                wall="east",
                rationale="Cooking owns the return leg.",
            ),
        ),
        rationale=("Test topology template.",),
    )


def item_walls(plan) -> dict[str, str]:
    return {
        assignment.item: assignment.wall
        for assignment in plan.item_assignments
    }


def test_zone_planning_assigns_core_workflow_items_to_walls() -> None:
    intent = StructuredIntent(required_items=["dishwasher", "hood"])

    plan = plan_zones_for_template(l_shape_template(), intent)

    walls = item_walls(plan)
    assert walls["sink"] == "north"
    assert walls["dishwasher"] == "north"
    assert walls["fridge"] == "north"
    assert walls["stove"] == "east"
    assert walls["hood"] == "east"
    assert walls["prep_base_cabinet"] == "east"
    assert plan.owner_for("dishwasher").zone_type == "cleaning"
    assert plan.owner_for("hood").zone_type == "cooking"


def test_zone_planning_assigns_run_roles_for_each_item() -> None:
    plan = plan_zones_for_template(
        l_shape_template(),
        StructuredIntent(required_items=["dishwasher", "hood"]),
    )

    assert plan.owner_for("sink").run_role == "primary"
    assert plan.owner_for("dishwasher").run_role == "primary"
    assert plan.owner_for("stove").run_role == "return"
    assert plan.owner_for("hood").run_role == "return"


def test_base_cabinets_only_suppresses_optional_storage() -> None:
    intent = StructuredIntent(
        base_cabinets_only=True,
        upper_cabinets=False,
        pantry_storage=True,
        tall_cabinets=True,
    )

    plan = plan_zones_for_template(l_shape_template(), intent)
    items = {assignment.item for assignment in plan.item_assignments}

    assert "base_cabinet" in items
    assert "wall_cabinet" not in items
    assert "tall_cabinet" not in items


def test_optional_storage_requests_add_upper_and_tall_ownership() -> None:
    intent = StructuredIntent(upper_cabinets=True, tall_cabinets=True)

    plan = plan_zones_for_template(l_shape_template(), intent)

    assert plan.owner_for("wall_cabinet").zone_type == "storage"
    assert plan.owner_for("tall_cabinet").zone_type == "storage"
    assert plan.owner_for("wall_cabinet").required is False
    assert plan.owner_for("tall_cabinet").required is False


def test_zone_plan_payload_is_serializable() -> None:
    plan = plan_zones_for_template(
        l_shape_template(),
        StructuredIntent(required_items=["dishwasher", "hood"]),
    )
    payload = plan.to_payload()

    assert payload["template_id"] == "template-l-test"
    assert payload["family"] == "L"
    assert len(payload["zones"]) == 5
    assert len(payload["item_assignments"]) >= 8
