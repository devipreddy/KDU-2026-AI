from __future__ import annotations

import json
from pathlib import Path

from auto_design.catalog.service import CatalogService
from auto_design.geometry import (
    AABB,
    DimensionsMM,
    PointMM,
    SpatialHashIndex,
    aabb_for_item,
    cells_for_aabb,
)
from auto_design.planner import (
    analyze_feasibility,
    generate_placement_plan,
    generate_topology_templates,
    plan_zones_for_template,
)
from auto_design.schemas import DesignInput, LayoutItem, StructuredIntent


ROOT = Path(__file__).resolve().parents[2]


def box(
    x: float,
    y: float,
    *,
    width: float = 600,
    depth: float = 600,
    height: float = 900,
) -> AABB:
    return AABB.from_center(
        PointMM(x, y, height / 2),
        DimensionsMM(width, depth, height),
    )


def placement_layout() -> dict[str, dict[str, object]]:
    payload = json.loads((ROOT / "input1.json").read_text(encoding="utf-8"))
    design_input = DesignInput.model_validate(payload)
    catalog = CatalogService.load(ROOT / "catalog.json")
    intent = StructuredIntent(
        layout_family="L",
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
    return placement.layout_payload()


def test_cells_for_aabb_uses_half_open_cell_boundaries() -> None:
    bounds = AABB(0, 0, 0, 600, 600, 600)

    assert cells_for_aabb(bounds, cell_size_mm=600) == ((0, 0, 0),)

    crossing = AABB(599, 599, 0, 601, 601, 10)
    assert set(cells_for_aabb(crossing, cell_size_mm=600)) == {
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    }


def test_spatial_index_tracks_occupancy_and_nearby_candidates() -> None:
    index = SpatialHashIndex(cell_size_mm=600)

    index.insert("a", box(300, 300))
    index.insert("b", box(900, 300))
    index.insert("far", box(5000, 5000))

    assert index.entry_count == 3
    assert index.occupied_cell_count >= 3
    assert index.ids_in_cell((0, 0, 0)) == ("a",)
    assert index.nearby_for_item("a") == ()
    assert [entry.item_id for entry in index.nearby_for_item("a", margin_mm=300)] == ["b"]


def test_collision_pairs_filter_broad_phase_candidates_to_real_intersections() -> None:
    index = SpatialHashIndex(cell_size_mm=600)

    index.insert("a", box(300, 300))
    index.insert("touching", box(900, 300))
    index.insert("overlap", box(590, 300, width=20))
    index.insert("far", box(3000, 3000))

    assert ("a", "far") not in index.candidate_pairs()
    assert [entry.item_id for entry in index.collisions_for_item("a")] == ["overlap"]
    assert [entry.item_id for entry in index.collisions_for_item("a", include_touching=True)] == [
        "overlap",
        "touching",
    ]
    assert {pair.second_id for pair in index.collision_pairs(include_touching=True)} >= {
        "overlap",
        "touching",
    }


def test_index_updates_replaced_and_removed_items() -> None:
    index = SpatialHashIndex(cell_size_mm=600)

    index.insert("moveable", box(300, 300))
    assert index.ids_in_cell((0, 0, 0)) == ("moveable",)

    index.insert("moveable", box(3000, 3000))
    assert index.ids_in_cell((0, 0, 0)) == ()
    assert "moveable" in index.ids_in_cell((4, 4, 0))

    index.remove("moveable")
    assert index.entry_count == 0
    assert index.occupied_cell_count == 0


def test_spatial_index_can_query_generated_placement_layout() -> None:
    layout = placement_layout()
    entries = []
    for key, item in layout.items():
        layout_item = LayoutItem.model_validate(item)
        entries.append((key, aabb_for_item(layout_item), layout_item))

    index = SpatialHashIndex.from_aabbs(entries, cell_size_mm=600)
    collisions = index.collision_pairs()
    query_item = LayoutItem.model_validate(layout["sink_north_1"])
    nearby = index.query_nearby(aabb_for_item(query_item), exclude_item_id="sink_north_1")

    assert index.entry_count == len(layout)
    assert all(pair.first_id != pair.second_id for pair in collisions)
    assert ("sink_north_1", "dishwasher_north_1") not in {
        (pair.first_id, pair.second_id) for pair in collisions
    }
    assert {entry.item_id for entry in nearby} >= {"dishwasher_north_1"}
