from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from auto_design.catalog.service import CatalogService
from auto_design.planner import (
    analyze_feasibility,
    generate_placement_plan,
    generate_topology_templates,
    plan_zones_for_template,
)
from auto_design.repair import repair_variant
from auto_design.schemas import DesignInput, StructuredIntent
from auto_design.schemas.environment import Environment
from auto_design.validation import validate_variant


ROOT = Path(__file__).resolve().parents[2]


def load_catalog() -> CatalogService:
    return CatalogService.load(ROOT / "catalog.json")


def generated_variant(
    name: str = "input1.json",
    *,
    family: str = "L",
) -> tuple[Environment, CatalogService, dict[str, object]]:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    design_input = DesignInput.model_validate(payload)
    catalog = load_catalog()
    intent = StructuredIntent(
        layout_family=family,
        required_items=design_input.preferences.must_have,
    )
    feasibility = analyze_feasibility(design_input, intent, catalog)
    template = generate_topology_templates(feasibility.to_payload())[0]
    zone_plan = plan_zones_for_template(template, intent)
    placement = generate_placement_plan(
        design_input.environment,
        template,
        zone_plan,
        catalog,
    )
    return design_input.environment, catalog, {
        "id": "variant-test",
        "family": template.family,
        "placement": placement.to_payload(),
        "layout": placement.layout_payload(),
    }


def all_items(variant: dict[str, object]) -> list[dict[str, Any]]:
    placement = variant["placement"]
    assert isinstance(placement, dict)
    items: list[dict[str, Any]] = []
    for run in placement["runs"]:
        assert isinstance(run, dict)
        items.extend(run["items"])
    items.extend(placement["overhead_items"])
    return items


def item_for(variant: dict[str, object], component: str) -> dict[str, Any]:
    item = next(item for item in all_items(variant) if item["component"] == component)
    assert isinstance(item, dict)
    return item


def rule_ids(variant: dict[str, object], environment: Environment) -> set[str]:
    return {
        violation.rule_id
        for violation in validate_variant(environment, variant).violations
    }


def action_ids(history: list[dict[str, object]]) -> set[str]:
    return {str(action["action"]) for action in history}


def test_repair_compacts_large_l_shape_work_triangle() -> None:
    environment, catalog, variant = generated_variant("input2.json")

    assert "WORKFLOW-03" in rule_ids(variant, environment)

    result = repair_variant(environment, catalog, variant)

    assert result.validation.violations == ()
    assert result.actions
    assert "compact_runs_to_shared_corner" in action_ids(
        result.variant["repair_history"]
    )
    assert result.variant["violations"] == []


def test_repair_realigns_hood_and_rebuilds_base_coverage() -> None:
    environment, catalog, valid_variant = generated_variant()
    variant = copy.deepcopy(valid_variant)
    hood = item_for(variant, "hood")
    placement = variant["placement"]
    assert isinstance(placement, dict)

    hood["position_mm"]["x"] += 500.0
    placement["base_coverage_valid"] = False
    placement["base_coverages"][0]["coverage_width_mm"] = 100.0
    placement["base_coverages"][0]["is_sufficient"] = False

    before = rule_ids(variant, environment)
    assert {"LAYOUT-02", "LAYOUT-04"} <= before

    result = repair_variant(environment, catalog, variant)
    after = {violation.rule_id for violation in result.validation.violations}

    assert {"LAYOUT-02", "LAYOUT-04"}.isdisjoint(after)
    assert {"realign_hood_over_stove", "rebuild_base_coverage"} <= action_ids(
        result.variant["repair_history"]
    )


def test_repair_inserts_base_filler_and_fixes_continuity() -> None:
    environment, catalog, valid_variant = generated_variant()
    variant = copy.deepcopy(valid_variant)
    placement = variant["placement"]
    assert isinstance(placement, dict)
    run = placement["runs"][0]
    assert isinstance(run, dict)
    items = run["items"]
    assert isinstance(items, list)

    items[2]["start_mm"] += 700.0
    items[2]["end_mm"] += 700.0
    run["continuity_gaps_mm"] = [700.0]
    run["is_continuous"] = False
    placement["is_continuous"] = False

    assert "LAYOUT-03" in rule_ids(variant, environment)

    result = repair_variant(environment, catalog, variant)
    repaired_keys = {item["key"] for item in all_items(result.variant)}

    assert "LAYOUT-03" not in {
        violation.rule_id
        for violation in result.validation.violations
    }
    assert "insert_base_and_fix_continuity" in action_ids(
        result.variant["repair_history"]
    )
    assert any(key.startswith("repair_filler_base") for key in repaired_keys)


def test_repair_snaps_fridge_and_items_out_of_door_swing() -> None:
    environment, catalog, valid_variant = generated_variant("input3.json")
    variant = copy.deepcopy(valid_variant)
    fridge = item_for(variant, "fridge")
    base = item_for(variant, "base_cabinet")

    fridge["position_mm"]["y"] = 500.0
    base["position_mm"] = {"x": 1050.0, "y": 450.0, "z": 450.0}

    before = rule_ids(variant, environment)
    assert {"NKBA-CL-01", "NKBA-CL-02"} <= before

    result = repair_variant(environment, catalog, variant)
    after = {violation.rule_id for violation in result.validation.violations}

    assert {"NKBA-CL-01", "NKBA-CL-02"}.isdisjoint(after)
    assert {"shift_fridge_to_corner", "avoid_door_and_window_openings"} <= action_ids(
        result.variant["repair_history"]
    )
