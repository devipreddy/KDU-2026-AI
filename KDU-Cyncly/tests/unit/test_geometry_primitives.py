from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_design.geometry import (
    AABB,
    DimensionsMM,
    PointMM,
    RoomBounds,
    aabb_for_item,
    normalize_rotation_z,
    rotation_for_anchor,
)
from auto_design.schemas import DesignInput, LayoutItem


ROOT = Path(__file__).resolve().parents[2]


def load_environment(name: str = "input3.json"):
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return DesignInput.model_validate(payload).environment


def test_room_bounds_are_derived_from_floor_and_wall_geometry() -> None:
    bounds = RoomBounds.from_environment(load_environment())

    assert bounds.min_x == 0
    assert bounds.min_y == 0
    assert bounds.min_z == 0
    assert bounds.max_x == 4200
    assert bounds.max_y == 3000
    assert bounds.max_z == 2700
    assert bounds.width == 4200
    assert bounds.depth == 3000
    assert bounds.height == 2700


def test_wall_coordinate_systems_match_renderer_anchor_rotations() -> None:
    bounds = RoomBounds.from_environment(load_environment())

    south = bounds.wall("south")
    north = bounds.wall("north")
    east = bounds.wall("east")
    west = bounds.wall("west")

    assert south.rotation_z_deg == 0
    assert north.rotation_z_deg == 180
    assert east.rotation_z_deg == 90
    assert west.rotation_z_deg == 270
    assert rotation_for_anchor("north") == 180


def test_wall_centers_place_objects_half_depth_inside_room() -> None:
    bounds = RoomBounds.from_environment(load_environment())

    assert bounds.wall("north").center_for(
        offset_mm=2100,
        depth_mm=600,
        height_mm=900,
    ) == PointMM(2100, 2700, 450)
    assert bounds.wall("south").center_for(
        offset_mm=2100,
        depth_mm=600,
        height_mm=900,
    ) == PointMM(2100, 300, 450)
    assert bounds.wall("east").center_for(
        offset_mm=1500,
        depth_mm=600,
        height_mm=900,
    ) == PointMM(3900, 1500, 450)
    assert bounds.wall("west").center_for(
        offset_mm=1500,
        depth_mm=600,
        height_mm=900,
    ) == PointMM(300, 1500, 450)


def test_wall_span_and_offset_bounds_are_symbolic_wall_coordinates() -> None:
    wall = RoomBounds.from_environment(load_environment()).wall("north")

    assert wall.span_for_center(center_offset_mm=2100, width_mm=600) == (1800, 2400)
    assert wall.is_offset_on_wall(0)
    assert wall.is_offset_on_wall(4200)
    assert not wall.is_offset_on_wall(4201)


def test_aabb_from_center_accounts_for_rotation_extents() -> None:
    center = PointMM(1000, 1000, 450)
    dimensions = DimensionsMM(width=600, depth=400, height=900)

    unrotated = AABB.from_center(center, dimensions, rotation_z_deg=0)
    rotated = AABB.from_center(center, dimensions, rotation_z_deg=90)

    assert unrotated.min_x == pytest.approx(700)
    assert unrotated.max_x == pytest.approx(1300)
    assert unrotated.min_y == pytest.approx(800)
    assert unrotated.max_y == pytest.approx(1200)
    assert rotated.min_x == pytest.approx(800)
    assert rotated.max_x == pytest.approx(1200)
    assert rotated.min_y == pytest.approx(700)
    assert rotated.max_y == pytest.approx(1300)
    assert rotated.min_z == pytest.approx(0)
    assert rotated.max_z == pytest.approx(900)


def test_aabb_intersection_touching_and_containment_rules() -> None:
    first = AABB(0, 0, 0, 100, 100, 100)
    touching = AABB(100, 0, 0, 200, 100, 100)
    overlapping = AABB(99, 0, 0, 200, 100, 100)

    assert not first.intersects(touching)
    assert first.intersects(touching, include_touching=True)
    assert first.intersects(overlapping)
    assert first.contains_point(PointMM(50, 50, 50))
    assert first.expanded(10).intersects(touching)


def test_room_bounds_contain_wall_anchored_object_aabb() -> None:
    bounds = RoomBounds.from_environment(load_environment())
    center = bounds.wall("north").center_for(offset_mm=2100, depth_mm=600, height_mm=900)
    item_bounds = AABB.from_center(center, DimensionsMM(600, 600, 900), 180)

    assert bounds.contains_aabb(item_bounds)


def test_aabb_for_layout_item_uses_layout_position_rotation_and_dimensions() -> None:
    item = LayoutItem.model_validate(
        {
            "product_id": "SKU-C01",
            "position_mm": {"x": 2100, "y": 2700, "z": 450},
            "dimensions_mm": {"width": 600, "depth": 600, "height": 900},
            "rotation_z_deg": 180,
            "anchor_wall": "north",
        }
    )

    bounds = aabb_for_item(item)

    assert bounds.min_x == pytest.approx(1800)
    assert bounds.max_x == pytest.approx(2400)
    assert bounds.min_y == pytest.approx(2400)
    assert bounds.max_y == pytest.approx(3000)


def test_rotation_normalization_wraps_degrees() -> None:
    assert normalize_rotation_z(360) == 0
    assert normalize_rotation_z(-90) == 270
    assert normalize_rotation_z(450) == 90
