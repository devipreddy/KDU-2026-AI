from __future__ import annotations

from typing import Literal

from pydantic import Field

from auto_design.schemas.common import (
    ContractModel,
    DimensionsMM,
    NonNegativeMM,
    Point3D,
    PositiveMM,
    WallDimensions,
)


WallAnchor = Literal["north", "south", "east", "west"]
OpeningKind = Literal["door", "window"]
HingeSide = Literal["left", "right"]
SwingDirection = Literal["in", "out", "sliding", "none"]


class Floor(ContractModel):
    points: list[Point3D] = Field(min_length=3)


class Wall(ContractModel):
    name: str = Field(min_length=1)
    dimensions: WallDimensions
    points: list[Point3D] = Field(min_length=2)
    anchor: WallAnchor | None = None
    thickness_mm: PositiveMM | None = None
    has_cabinets: bool | None = None


class Opening(ContractModel):
    id: str = Field(min_length=1)
    kind: OpeningKind
    wall: WallAnchor
    offset_mm: NonNegativeMM
    width_mm: PositiveMM
    height_mm: PositiveMM
    center_mm: Point3D
    dimensions_mm: DimensionsMM
    sill_mm: NonNegativeMM | None = None
    hinge_side: HingeSide | None = None
    swing_direction: SwingDirection | None = None


class Environment(ContractModel):
    floor: Floor
    wall: list[Wall] = Field(min_length=1)
    openings: list[Opening] = Field(default_factory=list)
