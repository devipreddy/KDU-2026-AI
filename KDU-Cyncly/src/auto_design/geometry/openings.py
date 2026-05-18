from __future__ import annotations

from dataclasses import dataclass

from auto_design.geometry.primitives import (
    AABB,
    DimensionsMM,
    PointMM,
    RoomBounds,
    rotation_for_anchor,
)
from auto_design.schemas.environment import Environment, Opening, OpeningKind, WallAnchor


DEFAULT_OPENING_FRONT_CLEARANCE_MM = 600.0
DEFAULT_DOOR_SWING_CLEARANCE_MM = 900.0


@dataclass(frozen=True)
class WallSpan:
    wall: WallAnchor
    start_mm: float
    end_mm: float
    opening_id: str
    kind: OpeningKind
    reason: str

    def __post_init__(self) -> None:
        if self.end_mm < self.start_mm:
            raise ValueError("WallSpan end_mm must be greater than or equal to start_mm.")

    @property
    def width_mm(self) -> float:
        return self.end_mm - self.start_mm

    @property
    def center_mm(self) -> float:
        return (self.start_mm + self.end_mm) / 2.0

    def overlaps(self, start_mm: float, end_mm: float, *, allowance_mm: float = 0.0) -> bool:
        if end_mm < start_mm:
            raise ValueError("Span end_mm must be greater than or equal to start_mm.")
        return not (
            end_mm <= self.start_mm + allowance_mm
            or start_mm >= self.end_mm - allowance_mm
        )

    def expanded(self, margin_mm: float) -> WallSpan:
        return WallSpan(
            wall=self.wall,
            start_mm=self.start_mm - margin_mm,
            end_mm=self.end_mm + margin_mm,
            opening_id=self.opening_id,
            kind=self.kind,
            reason=self.reason,
        )


@dataclass(frozen=True)
class OpeningConstraint:
    opening_id: str
    kind: OpeningKind
    wall: WallAnchor
    wall_span: WallSpan
    opening_aabb: AABB
    front_clearance_aabb: AABB
    door_swing_aabb: AABB | None = None

    @property
    def blocks_placement(self) -> bool:
        return True

    @property
    def reservation_aabbs(self) -> tuple[AABB, ...]:
        if self.door_swing_aabb is None:
            return (self.front_clearance_aabb,)
        return (self.front_clearance_aabb, self.door_swing_aabb)


@dataclass(frozen=True)
class OpeningConstraintSet:
    constraints: tuple[OpeningConstraint, ...]

    @property
    def blocked_spans(self) -> tuple[WallSpan, ...]:
        return tuple(constraint.wall_span for constraint in self.constraints)

    @property
    def door_swing_reservations(self) -> tuple[AABB, ...]:
        return tuple(
            constraint.door_swing_aabb
            for constraint in self.constraints
            if constraint.door_swing_aabb is not None
        )

    @property
    def placement_reservations(self) -> tuple[AABB, ...]:
        return tuple(
            reservation
            for constraint in self.constraints
            for reservation in constraint.reservation_aabbs
        )

    def for_wall(self, wall: WallAnchor) -> tuple[OpeningConstraint, ...]:
        return tuple(constraint for constraint in self.constraints if constraint.wall == wall)

    def blocked_spans_for_wall(self, wall: WallAnchor) -> tuple[WallSpan, ...]:
        return tuple(span for span in self.blocked_spans if span.wall == wall)

    def blocked_spans_by_wall(self) -> dict[WallAnchor, tuple[WallSpan, ...]]:
        walls: tuple[WallAnchor, ...] = ("north", "south", "east", "west")
        return {wall: self.blocked_spans_for_wall(wall) for wall in walls}

    def blocking_openings_for_span(
        self,
        wall: WallAnchor,
        start_mm: float,
        end_mm: float,
        *,
        allowance_mm: float = 0.0,
    ) -> tuple[OpeningConstraint, ...]:
        return tuple(
            constraint
            for constraint in self.for_wall(wall)
            if constraint.wall_span.overlaps(
                start_mm,
                end_mm,
                allowance_mm=allowance_mm,
            )
        )

    def is_span_blocked(
        self,
        wall: WallAnchor,
        start_mm: float,
        end_mm: float,
        *,
        allowance_mm: float = 0.0,
    ) -> bool:
        return bool(
            self.blocking_openings_for_span(
                wall,
                start_mm,
                end_mm,
                allowance_mm=allowance_mm,
            )
        )


def opening_wall_span(opening: Opening) -> WallSpan:
    return WallSpan(
        wall=opening.wall,
        start_mm=float(opening.offset_mm),
        end_mm=float(opening.offset_mm + opening.width_mm),
        opening_id=opening.id,
        kind=opening.kind,
        reason=f"{opening.kind} opening blocks wall placement",
    )


def opening_physical_aabb(opening: Opening) -> AABB:
    return AABB.from_center(
        center=PointMM.from_schema(opening.center_mm),
        dimensions=DimensionsMM.from_schema(opening.dimensions_mm),
        rotation_z_deg=rotation_for_anchor(opening.wall),
    )


def front_clearance_aabb(
    opening: Opening,
    bounds: RoomBounds,
    *,
    depth_mm: float = DEFAULT_OPENING_FRONT_CLEARANCE_MM,
) -> AABB:
    wall = bounds.wall(opening.wall)
    span = opening_wall_span(opening)
    z_center = float(opening.center_mm.z)
    return AABB.from_center(
        center=wall.center_for(
            offset_mm=span.center_mm,
            depth_mm=depth_mm,
            height_mm=float(opening.height_mm),
            z_center_mm=z_center,
        ),
        dimensions=DimensionsMM(
            width=span.width_mm,
            depth=depth_mm,
            height=float(opening.height_mm),
        ),
        rotation_z_deg=wall.rotation_z_deg,
    )


def door_swing_aabb(
    opening: Opening,
    bounds: RoomBounds,
    *,
    clearance_mm: float = DEFAULT_DOOR_SWING_CLEARANCE_MM,
) -> AABB | None:
    if opening.kind != "door" or opening.swing_direction != "in":
        return None

    wall = bounds.wall(opening.wall)
    span = opening_wall_span(opening)
    reservation_width = clearance_mm
    if opening.hinge_side == "right":
        start = span.end_mm - reservation_width
    else:
        start = span.start_mm
    start = min(max(start, 0.0), max(wall.length_mm - reservation_width, 0.0))
    center_offset = start + (reservation_width / 2.0)
    return AABB.from_center(
        center=wall.center_for(
            offset_mm=center_offset,
            depth_mm=clearance_mm,
            height_mm=float(opening.height_mm),
            z_center_mm=float(opening.height_mm) / 2.0,
        ),
        dimensions=DimensionsMM(
            width=reservation_width,
            depth=clearance_mm,
            height=float(opening.height_mm),
        ),
        rotation_z_deg=wall.rotation_z_deg,
    )


def opening_constraint_for(
    opening: Opening,
    bounds: RoomBounds,
    *,
    front_depth_mm: float = DEFAULT_OPENING_FRONT_CLEARANCE_MM,
    door_swing_clearance_mm: float = DEFAULT_DOOR_SWING_CLEARANCE_MM,
) -> OpeningConstraint:
    return OpeningConstraint(
        opening_id=opening.id,
        kind=opening.kind,
        wall=opening.wall,
        wall_span=opening_wall_span(opening),
        opening_aabb=opening_physical_aabb(opening),
        front_clearance_aabb=front_clearance_aabb(
            opening,
            bounds,
            depth_mm=front_depth_mm,
        ),
        door_swing_aabb=door_swing_aabb(
            opening,
            bounds,
            clearance_mm=door_swing_clearance_mm,
        ),
    )


def build_opening_constraints(
    environment: Environment,
    *,
    front_depth_mm: float = DEFAULT_OPENING_FRONT_CLEARANCE_MM,
    door_swing_clearance_mm: float = DEFAULT_DOOR_SWING_CLEARANCE_MM,
) -> OpeningConstraintSet:
    bounds = RoomBounds.from_environment(environment)
    return OpeningConstraintSet(
        constraints=tuple(
            opening_constraint_for(
                opening,
                bounds,
                front_depth_mm=front_depth_mm,
                door_swing_clearance_mm=door_swing_clearance_mm,
            )
            for opening in environment.openings
        )
    )
