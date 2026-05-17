from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_design.catalog.service import CatalogService
from auto_design.planner import (
    MIN_PREP_COUNTER_MM,
    allowed_wall_runs,
    analyze_feasibility,
    required_appliance_footprint_mm,
    required_footprint_items,
)
from auto_design.schemas import DesignInput, StructuredIntent


ROOT = Path(__file__).resolve().parents[2]


def load_design_input(name: str = "input1.json") -> DesignInput:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return DesignInput.model_validate(payload)


def load_catalog() -> CatalogService:
    return CatalogService.load(ROOT / "catalog.json")


def compact_room_payload(
    *,
    width_mm: int,
    depth_mm: int,
    cabinet_walls: set[str],
    prompt: str = "",
) -> dict[str, object]:
    wall_specs = [
        ("south_wall", "south", width_mm),
        ("north_wall", "north", width_mm),
        ("east_wall", "east", depth_mm),
        ("west_wall", "west", depth_mm),
    ]
    points_by_anchor = {
        "south": [
            {"x": 0, "y": 0, "z": 0},
            {"x": width_mm, "y": 0, "z": 0},
            {"x": width_mm, "y": 0, "z": 2700},
            {"x": 0, "y": 0, "z": 2700},
        ],
        "north": [
            {"x": 0, "y": depth_mm, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 2700},
            {"x": 0, "y": depth_mm, "z": 2700},
        ],
        "east": [
            {"x": width_mm, "y": 0, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 2700},
            {"x": width_mm, "y": 0, "z": 2700},
        ],
        "west": [
            {"x": 0, "y": 0, "z": 0},
            {"x": 0, "y": depth_mm, "z": 0},
            {"x": 0, "y": depth_mm, "z": 2700},
            {"x": 0, "y": 0, "z": 2700},
        ],
    }
    return {
        "environment": {
            "floor": {
                "points": [
                    {"x": 0, "y": 0, "z": 0},
                    {"x": width_mm, "y": 0, "z": 0},
                    {"x": width_mm, "y": depth_mm, "z": 0},
                    {"x": 0, "y": depth_mm, "z": 0},
                ]
            },
            "wall": [
                {
                    "name": name,
                    "anchor": anchor,
                    "thickness_mm": 100,
                    "has_cabinets": anchor in cabinet_walls,
                    "dimensions": {"length_mm": length, "height": 2700},
                    "points": points_by_anchor[anchor],
                }
                for name, anchor, length in wall_specs
            ],
            "openings": [],
        },
        "preferences": {
            "budget_tier": "mid",
            "must_have": ["dishwasher", "hood"],
            "avoid": [],
            "prompt": prompt,
            "catalog": "./catalog.json",
        },
    }


def test_openings_reduce_available_wall_runs() -> None:
    runs = {
        run.anchor: run
        for run in allowed_wall_runs(load_design_input("input3.json").environment)
    }

    assert runs["north"].available_segments_mm == ((0.0, 1500.0), (2700.0, 4200.0))
    assert runs["east"].available_segments_mm == ((0.0, 600.0), (1400.0, 3000.0))
    assert "south" not in runs
    assert "west" not in runs


def test_required_appliance_footprint_uses_catalog_widths() -> None:
    intent = StructuredIntent(required_items=["dishwasher", "hood"])
    items = required_footprint_items(load_catalog(), intent)
    footprint = required_appliance_footprint_mm(items)

    assert [item.kind for item in items] == ["sink", "stove", "fridge", "dishwasher", "hood"]
    assert footprint == pytest.approx(2350)
    assert footprint + MIN_PREP_COUNTER_MM == pytest.approx(2950)
    assert next(item for item in items if item.kind == "dishwasher").product_id == "SKU-A08"
    assert next(item for item in items if item.kind == "hood").counted_in_run is False


def test_feasibility_selects_l_shape_when_two_adjacent_walls_fit() -> None:
    design_input = load_design_input("input1.json")
    intent = StructuredIntent(required_items=design_input.preferences.must_have)

    result = analyze_feasibility(design_input, intent, load_catalog())

    assert result.status == "feasible"
    assert result.feasible is True
    assert result.requested_family is None
    assert result.selected_family == "L"
    assert result.topology_fits["L"].feasible is True
    assert result.topology_fits["U"].feasible is False


def test_requested_u_shape_falls_back_when_only_two_walls_are_available() -> None:
    design_input = load_design_input("input1.json")
    intent = StructuredIntent(
        layout_family="U",
        required_items=design_input.preferences.must_have,
    )

    result = analyze_feasibility(design_input, intent, load_catalog())
    payload = result.to_payload()

    assert result.status == "fallback"
    assert result.feasible is True
    assert result.requested_family_feasible is False
    assert result.selected_family == "L"
    assert result.fallback_family == "L"
    assert "Requested U layout rejected" in result.fallback_reason
    assert payload["requested_family_feasible"] is False
    assert payload["topology_fits"]["U"]["feasible"] is False


def test_requested_i_shape_rejects_room_with_insufficient_run_and_walkway() -> None:
    design_input = DesignInput.model_validate(
        compact_room_payload(
            width_mm=1600,
            depth_mm=1500,
            cabinet_walls={"north"},
            prompt="single-wall kitchen",
        )
    )
    intent = StructuredIntent(
        layout_family="I",
        required_items=design_input.preferences.must_have,
    )

    result = analyze_feasibility(design_input, intent, load_catalog())

    assert result.status == "infeasible"
    assert result.feasible is False
    assert result.selected_family is None
    assert result.requested_family_feasible is False
    assert any("continuous run" in reason for reason in result.topology_fits["I"].reasons)
    assert any("walkway clearance" in reason for reason in result.topology_fits["I"].reasons)
