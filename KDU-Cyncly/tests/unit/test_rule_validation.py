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
from auto_design.schemas import DesignInput, StructuredIntent
from auto_design.schemas.environment import Environment
from auto_design.validation import RuleViolation, validate_variant


ROOT = Path(__file__).resolve().parents[2]


def generated_variant(
    name: str = "input1.json",
    *,
    family: str = "L",
    required_items: list[str] | None = None,
) -> tuple[Environment, dict[str, object]]:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    design_input = DesignInput.model_validate(payload)
    catalog = CatalogService.load(ROOT / "catalog.json")
    intent = StructuredIntent(
        layout_family=family,
        required_items=required_items or design_input.preferences.must_have,
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
    return design_input.environment, {
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
    item = next(
        item
        for item in all_items(variant)
        if item["component"] == component
    )
    assert isinstance(item, dict)
    return item


def rule_ids(violations: tuple[RuleViolation, ...]) -> set[str]:
    return {violation.rule_id for violation in violations}


def test_generated_layout_returns_structured_rule_pack_results() -> None:
    environment, variant = generated_variant()
    result = validate_variant(environment, variant)
    payloads = [violation.to_payload() for violation in result.violations]

    assert result.variant_id == "variant-test"
    assert all({"rule_id", "severity", "text"} <= set(payload) for payload in payloads)
    assert all(payload["text"] for payload in payloads)
    assert "WORKFLOW-03" in rule_ids(result.violations)
    assert not (
        {"NKBA-CL-01", "NKBA-CL-02", "LAYOUT-03", "LAYOUT-04", "LAYOUT-05"}
        & rule_ids(result.violations)
    )


def test_nkba_rules_report_fridge_and_door_swing_clearance_violations() -> None:
    environment, valid_variant = generated_variant("input3.json")
    variant = copy.deepcopy(valid_variant)
    base_item = item_for(variant, "base_cabinet")
    fridge = item_for(variant, "fridge")

    base_item["position_mm"] = {"x": 1050.0, "y": 450.0, "z": 450.0}
    fridge["position_mm"]["y"] = 500.0

    result = validate_variant(environment, variant)

    assert {"NKBA-CL-01", "NKBA-CL-02"} <= rule_ids(result.violations)
    assert {
        violation.severity
        for violation in result.violations
        if violation.rule_id in {"NKBA-CL-01", "NKBA-CL-02"}
    } == {"hard"}


def test_workflow_rules_report_adjacency_spacing_and_triangle_violations() -> None:
    environment, valid_variant = generated_variant()
    variant = copy.deepcopy(valid_variant)
    sink = item_for(variant, "sink")
    dishwasher = item_for(variant, "dishwasher")
    fridge = item_for(variant, "fridge")
    stove = item_for(variant, "stove")

    sink["position_mm"]["x"] += 10000.0
    dishwasher["position_mm"]["x"] += 5000.0
    fridge["position_mm"] = dict(stove["position_mm"])

    result = validate_variant(environment, variant)

    assert {"WORKFLOW-01", "WORKFLOW-02", "WORKFLOW-03"} <= rule_ids(
        result.violations
    )
    assert {
        violation.severity
        for violation in result.violations
        if violation.rule_id.startswith("WORKFLOW")
    } == {"soft"}


def test_layout_alignment_and_continuity_rules_report_violations() -> None:
    environment, valid_variant = generated_variant("input3.json")
    variant = copy.deepcopy(valid_variant)
    placement = variant["placement"]
    assert isinstance(placement, dict)
    runs = placement["runs"]
    assert isinstance(runs, list)
    hood = item_for(variant, "hood")

    placement["is_continuous"] = False
    runs[0]["continuity_gaps_mm"] = [100.0]
    runs[0]["starts_with_base"] = False
    hood["position_mm"]["x"] += 500.0

    result = validate_variant(environment, variant)

    assert {"LAYOUT-01", "LAYOUT-02", "LAYOUT-03", "LAYOUT-05"} <= rule_ids(
        result.violations
    )
    assert {
        violation.severity
        for violation in result.violations
        if violation.rule_id in {"LAYOUT-03", "LAYOUT-05"}
    } == {"hard"}


def test_base_coverage_and_corner_rules_detect_invalid_payloads() -> None:
    environment, valid_variant = generated_variant()
    variant = copy.deepcopy(valid_variant)
    placement = variant["placement"]
    assert isinstance(placement, dict)
    first_coverage = placement["base_coverages"][0]
    fridge = item_for(variant, "fridge")

    placement["base_coverage_valid"] = False
    first_coverage["is_sufficient"] = False
    first_coverage["coverage_width_mm"] = 100.0
    fridge["start_mm"] = 1700.0
    fridge["end_mm"] = 2400.0

    result = validate_variant(environment, variant)

    assert {"LAYOUT-04", "LAYOUT-06"} <= rule_ids(result.violations)
    assert next(
        violation.severity
        for violation in result.violations
        if violation.rule_id == "LAYOUT-04"
    ) == "hard"
