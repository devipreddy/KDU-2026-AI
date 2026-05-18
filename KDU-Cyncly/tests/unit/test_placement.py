from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_design.catalog.service import CatalogService
from auto_design.planner import (
    analyze_feasibility,
    generate_placement_plan,
    generate_topology_templates,
    plan_zones_for_template,
)
from auto_design.schemas import DesignInput, LayoutItem, StructuredIntent


ROOT = Path(__file__).resolve().parents[2]


def load_design_input(name: str = "input1.json") -> DesignInput:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return DesignInput.model_validate(payload)


def load_catalog() -> CatalogService:
    return CatalogService.load(ROOT / "catalog.json")


def placement_plan_for(
    family: str | None = "L",
    name: str = "input1.json",
    required_items: list[str] | None = None,
):
    design_input = load_design_input(name)
    catalog = load_catalog()
    intent = StructuredIntent(
        layout_family=family,
        required_items=required_items or design_input.preferences.must_have,
    )
    feasibility = analyze_feasibility(design_input, intent, catalog)
    template = generate_topology_templates(feasibility.to_payload())[0]
    zone_plan = plan_zones_for_template(template, intent)
    return generate_placement_plan(design_input.environment, template, zone_plan, catalog)


def test_placement_generates_continuous_snapped_wall_runs() -> None:
    plan = placement_plan_for("L")

    assert plan.is_continuous
    for run in plan.runs:
        assert run.is_continuous
        assert all(gap == pytest.approx(0.0) for gap in run.continuity_gaps_mm)
        assert run.base_items[0].start_mm == pytest.approx(run.start_mm)
        for index, item in enumerate(run.base_items[:-1]):
            assert item.end_mm == pytest.approx(run.base_items[index + 1].start_mm)
        assert all(item.wall == run.wall for item in run.base_items)
        assert all(item.layer == "base_run" for item in run.base_items)


def test_every_run_starts_and_ends_with_base_cabinets() -> None:
    plan = placement_plan_for("L")

    for run in plan.runs:
        assert run.starts_with_base
        assert run.ends_with_base
        assert run.base_items[0].is_base_cabinet
        assert run.base_items[-1].is_base_cabinet
        assert run.base_items[0].is_terminator or run.base_items[0].component in {
            "base_cabinet",
            "prep_base_cabinet",
        }
        assert run.base_items[-1].is_terminator or run.base_items[-1].component in {
            "base_cabinet",
            "prep_base_cabinet",
        }


def test_appliances_and_sinks_have_equal_or_larger_base_backing() -> None:
    plan = placement_plan_for("L")
    backed_components = {"dishwasher", "fridge", "hood", "sink", "stove"}

    assert plan.base_coverage_valid
    for item in plan.items:
        if item.component not in backed_components:
            continue
        assert item.backed_by_product_id is not None
        assert item.backing_width_mm is not None
        assert item.backing_width_mm >= item.dimensions_mm.width


def test_explicit_base_coverage_records_cover_every_backed_item() -> None:
    plan = placement_plan_for("L")
    coverage_by_item = {
        coverage.covered_item_key: coverage
        for coverage in plan.base_coverages
    }

    backed_items = [
        item for item in plan.items
        if item.component in {"dishwasher", "fridge", "hood", "sink", "stove"}
    ]

    assert backed_items
    assert {item.key for item in backed_items}.issubset(coverage_by_item)
    for item in backed_items:
        coverage = coverage_by_item[item.key]
        assert coverage.wall == item.wall
        assert coverage.run_role == item.run_role
        assert coverage.covered_product_id == item.product_id
        assert coverage.coverage_width_mm >= item.dimensions_mm.width
        assert coverage.is_sufficient


def test_requested_oven_gets_cooking_placement_and_base_coverage() -> None:
    plan = placement_plan_for("L", required_items=["dishwasher", "hood", "oven"])
    ovens = [item for item in plan.items if item.component == "oven"]
    coverage_by_item = {
        coverage.covered_item_key: coverage
        for coverage in plan.base_coverages
    }

    assert len(ovens) == 1
    assert ovens[0].zone_type == "cooking"
    assert ovens[0].backed_by_product_id is not None
    assert ovens[0].key in coverage_by_item
    assert coverage_by_item[ovens[0].key].kind == "below"


def test_base_coverage_payload_is_serializable_and_explicit() -> None:
    payload = placement_plan_for("L").to_payload()

    assert payload["base_coverage_valid"] is True
    assert payload["base_coverages"]
    first = payload["base_coverages"][0]
    assert first["covered_item_key"]
    assert first["base_product_id"].startswith("SKU-C")
    assert first["coverage_width_mm"] >= first["covered_width_mm"]


def test_layout_payload_uses_renderer_compatible_layout_items_and_known_skus() -> None:
    catalog = load_catalog()
    layout = placement_plan_for("L").layout_payload()

    assert layout
    for item in layout.values():
        validated = LayoutItem.model_validate(item)
        assert validated.product_id is not None
        assert catalog.has_sku(validated.product_id)
        assert validated.anchor_wall in {"north", "south", "east", "west"}


def test_single_wall_placement_compacts_optional_fillers_to_fit_run() -> None:
    plan = placement_plan_for("I")
    run = plan.runs[0]
    components = {item.component for item in run.base_items}

    assert run.end_mm <= 3600
    assert run.is_continuous
    assert run.starts_with_base
    assert run.ends_with_base
    assert {"dishwasher", "fridge", "sink", "stove"}.issubset(components)
    assert "prep_base_cabinet" not in components
