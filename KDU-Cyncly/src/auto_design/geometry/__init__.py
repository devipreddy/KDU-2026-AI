"""Deterministic spatial geometry package."""

from auto_design.geometry.openings import (
    DEFAULT_DOOR_SWING_CLEARANCE_MM,
    DEFAULT_OPENING_FRONT_CLEARANCE_MM,
    OpeningConstraint,
    OpeningConstraintSet,
    WallSpan,
    build_opening_constraints,
    door_swing_aabb,
    front_clearance_aabb,
    opening_constraint_for,
    opening_physical_aabb,
    opening_wall_span,
)
from auto_design.geometry.primitives import (
    AABB,
    DimensionsMM,
    PointMM,
    RoomBounds,
    WallCoordinateSystem,
    aabb_for_item,
    normalize_rotation_z,
    rotation_for_anchor,
)

__all__ = [
    "AABB",
    "DEFAULT_DOOR_SWING_CLEARANCE_MM",
    "DEFAULT_OPENING_FRONT_CLEARANCE_MM",
    "DimensionsMM",
    "OpeningConstraint",
    "OpeningConstraintSet",
    "PointMM",
    "RoomBounds",
    "WallSpan",
    "WallCoordinateSystem",
    "aabb_for_item",
    "build_opening_constraints",
    "door_swing_aabb",
    "front_clearance_aabb",
    "normalize_rotation_z",
    "opening_constraint_for",
    "opening_physical_aabb",
    "opening_wall_span",
    "rotation_for_anchor",
]
