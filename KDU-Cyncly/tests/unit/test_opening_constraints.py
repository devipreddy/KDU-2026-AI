from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_design.geometry import (
    RoomBounds,
    WallSpan,
    build_opening_constraints,
    door_swing_aabb,
    front_clearance_aabb,
    opening_physical_aabb,
)
from auto_design.schemas import DesignInput


ROOT = Path(__file__).resolve().parents[2]


def load_environment(name: str = "input3.json"):
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return DesignInput.model_validate(payload).environment


def test_openings_become_blocked_wall_spans() -> None:
    constraints = build_opening_constraints(load_environment())

    assert len(constraints.constraints) == 3
    assert constraints.blocked_spans_for_wall("south")[0].opening_id == "south_door"
    assert constraints.blocked_spans_for_wall("south")[0].start_mm == 600
    assert constraints.blocked_spans_for_wall("south")[0].end_mm == 1500
    assert constraints.blocked_spans_for_wall("north")[0].opening_id == "north_window"
    assert constraints.blocked_spans_for_wall("north")[0].start_mm == 1500
    assert constraints.blocked_spans_for_wall("north")[0].end_mm == 2700
    assert constraints.blocked_spans_for_wall("east")[0].opening_id == "east_window"
    assert constraints.blocked_spans_for_wall("east")[0].start_mm == 600
    assert constraints.blocked_spans_for_wall("east")[0].end_mm == 1400


def test_span_blocking_queries_report_overlapping_openings() -> None:
    constraints = build_opening_constraints(load_environment())

    assert constraints.is_span_blocked("north", 1800, 2400)
    assert not constraints.is_span_blocked("north", 3000, 3600)

    blocked = constraints.blocking_openings_for_span("north", 1800, 2400)
    assert [constraint.opening_id for constraint in blocked] == ["north_window"]


def test_blocked_spans_are_grouped_by_wall_anchor() -> None:
    spans = build_opening_constraints(load_environment()).blocked_spans_by_wall()

    assert set(spans) == {"north", "south", "east", "west"}
    assert [span.opening_id for span in spans["north"]] == ["north_window"]
    assert [span.opening_id for span in spans["south"]] == ["south_door"]
    assert [span.opening_id for span in spans["east"]] == ["east_window"]
    assert spans["west"] == ()


def test_inward_door_reserves_swing_clearance_inside_room() -> None:
    constraints = build_opening_constraints(load_environment())

    assert len(constraints.door_swing_reservations) == 1
    reservation = constraints.door_swing_reservations[0]

    assert reservation.min_x == pytest.approx(600)
    assert reservation.max_x == pytest.approx(1500)
    assert reservation.min_y == pytest.approx(0)
    assert reservation.max_y == pytest.approx(900)
    assert reservation.min_z == pytest.approx(0)
    assert reservation.max_z == pytest.approx(2100)


def test_door_swing_reservation_uses_rule_clearance_not_door_width() -> None:
    environment = load_environment()
    south_door = next(
        opening for opening in environment.openings if opening.id == "south_door"
    )
    narrow_door = south_door.model_copy(update={"width_mm": 800})

    reservation = door_swing_aabb(
        narrow_door,
        RoomBounds.from_environment(environment),
    )

    assert reservation is not None
    assert reservation.width == pytest.approx(900)
    assert reservation.depth == pytest.approx(900)


def test_window_front_clearance_is_reserved_from_wall_span() -> None:
    environment = load_environment()
    north_window = next(
        opening for opening in environment.openings if opening.id == "north_window"
    )
    clearance = front_clearance_aabb(
        north_window,
        RoomBounds.from_environment(environment),
    )

    assert clearance.min_x == pytest.approx(1500)
    assert clearance.max_x == pytest.approx(2700)
    assert clearance.min_y == pytest.approx(2400)
    assert clearance.max_y == pytest.approx(3000)
    assert clearance.min_z == pytest.approx(900)
    assert clearance.max_z == pytest.approx(2100)


def test_opening_physical_aabb_respects_wall_rotation() -> None:
    environment = load_environment()
    east_window = next(
        opening for opening in environment.openings if opening.id == "east_window"
    )
    bounds = opening_physical_aabb(east_window)

    assert bounds.min_x == pytest.approx(4150)
    assert bounds.max_x == pytest.approx(4250)
    assert bounds.min_y == pytest.approx(600)
    assert bounds.max_y == pytest.approx(1400)
    assert bounds.min_z == pytest.approx(900)
    assert bounds.max_z == pytest.approx(2100)


def test_empty_opening_list_creates_empty_constraint_set() -> None:
    constraints = build_opening_constraints(load_environment("input1.json"))

    assert constraints.constraints == ()
    assert constraints.blocked_spans == ()
    assert constraints.door_swing_reservations == ()
    assert constraints.placement_reservations == ()


def test_wall_span_overlap_allows_touching_but_blocks_intersection() -> None:
    span = WallSpan(
        wall="north",
        start_mm=1000,
        end_mm=1600,
        opening_id="north_window",
        kind="window",
        reason="window opening blocks wall placement",
    )

    assert span.overlaps(1200, 1400)
    assert not span.overlaps(1600, 2200)
    assert not span.overlaps(1590, 2200, allowance_mm=10)
