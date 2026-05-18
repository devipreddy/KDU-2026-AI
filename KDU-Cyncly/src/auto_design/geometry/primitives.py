from __future__ import annotations

import math
from dataclasses import dataclass

from auto_design.schemas.common import DimensionsMM as SchemaDimensionsMM
from auto_design.schemas.common import Point3D
from auto_design.schemas.environment import Environment, WallAnchor
from auto_design.schemas.output import LayoutItem


@dataclass(frozen=True)
class PointMM:
    x: float
    y: float
    z: float = 0.0

    @classmethod
    def from_schema(cls, point: Point3D) -> PointMM:
        return cls(x=float(point.x), y=float(point.y), z=float(point.z))

    def translate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> PointMM:
        return PointMM(self.x + dx, self.y + dy, self.z + dz)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class DimensionsMM:
    width: float
    depth: float
    height: float

    @classmethod
    def from_schema(cls, dimensions: SchemaDimensionsMM) -> DimensionsMM:
        return cls(
            width=float(dimensions.width),
            depth=float(dimensions.depth),
            height=float(dimensions.height),
        )

    def to_dict(self) -> dict[str, float]:
        return {"width": self.width, "depth": self.depth, "height": self.height}


@dataclass(frozen=True)
class AABB:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @classmethod
    def from_center(
        cls,
        center: PointMM,
        dimensions: DimensionsMM,
        rotation_z_deg: float = 0.0,
    ) -> AABB:
        radians = math.radians(normalize_rotation_z(rotation_z_deg))
        half_width = dimensions.width / 2.0
        half_depth = dimensions.depth / 2.0
        half_x = (abs(math.cos(radians)) * half_width) + (
            abs(math.sin(radians)) * half_depth
        )
        half_y = (abs(math.sin(radians)) * half_width) + (
            abs(math.cos(radians)) * half_depth
        )
        half_z = dimensions.height / 2.0
        return cls(
            min_x=center.x - half_x,
            min_y=center.y - half_y,
            min_z=center.z - half_z,
            max_x=center.x + half_x,
            max_y=center.y + half_y,
            max_z=center.z + half_z,
        )

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def center(self) -> PointMM:
        return PointMM(
            x=(self.min_x + self.max_x) / 2.0,
            y=(self.min_y + self.max_y) / 2.0,
            z=(self.min_z + self.max_z) / 2.0,
        )

    def intersects(self, other: AABB, *, include_touching: bool = False) -> bool:
        if include_touching:
            return not (
                self.max_x < other.min_x
                or self.min_x > other.max_x
                or self.max_y < other.min_y
                or self.min_y > other.max_y
                or self.max_z < other.min_z
                or self.min_z > other.max_z
            )
        return not (
            self.max_x <= other.min_x
            or self.min_x >= other.max_x
            or self.max_y <= other.min_y
            or self.min_y >= other.max_y
            or self.max_z <= other.min_z
            or self.min_z >= other.max_z
        )

    def contains_point(self, point: PointMM) -> bool:
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
            and self.min_z <= point.z <= self.max_z
        )

    def contains_aabb(self, other: AABB) -> bool:
        return (
            self.min_x <= other.min_x
            and self.max_x >= other.max_x
            and self.min_y <= other.min_y
            and self.max_y >= other.max_y
            and self.min_z <= other.min_z
            and self.max_z >= other.max_z
        )

    def expanded(self, margin_mm: float) -> AABB:
        return AABB(
            min_x=self.min_x - margin_mm,
            min_y=self.min_y - margin_mm,
            min_z=self.min_z - margin_mm,
            max_x=self.max_x + margin_mm,
            max_y=self.max_y + margin_mm,
            max_z=self.max_z + margin_mm,
        )


@dataclass(frozen=True)
class RoomBounds:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @classmethod
    def from_environment(cls, environment: Environment) -> RoomBounds:
        floor_points = [PointMM.from_schema(point) for point in environment.floor.points]
        wall_points = [
            PointMM.from_schema(point)
            for wall in environment.wall
            for point in wall.points
        ]
        all_points = [*floor_points, *wall_points]
        wall_heights = [wall.dimensions.height for wall in environment.wall]
        max_height = max([point.z for point in all_points], default=0.0)
        if wall_heights:
            max_height = max(max_height, max(float(height) for height in wall_heights))

        return cls(
            min_x=min(point.x for point in floor_points),
            min_y=min(point.y for point in floor_points),
            min_z=min(point.z for point in all_points),
            max_x=max(point.x for point in floor_points),
            max_y=max(point.y for point in floor_points),
            max_z=max_height,
        )

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def aabb(self) -> AABB:
        return AABB(
            min_x=self.min_x,
            min_y=self.min_y,
            min_z=self.min_z,
            max_x=self.max_x,
            max_y=self.max_y,
            max_z=self.max_z,
        )

    def wall(self, anchor: WallAnchor) -> WallCoordinateSystem:
        return WallCoordinateSystem.from_bounds(anchor, self)

    def contains_aabb(self, bounds: AABB) -> bool:
        return self.aabb.contains_aabb(bounds)


@dataclass(frozen=True)
class WallCoordinateSystem:
    anchor: WallAnchor
    origin: PointMM
    tangent: tuple[float, float]
    inward_normal: tuple[float, float]
    length_mm: float
    rotation_z_deg: float

    @classmethod
    def from_bounds(cls, anchor: WallAnchor, bounds: RoomBounds) -> WallCoordinateSystem:
        if anchor == "south":
            return cls(
                anchor=anchor,
                origin=PointMM(bounds.min_x, bounds.min_y, bounds.min_z),
                tangent=(1.0, 0.0),
                inward_normal=(0.0, 1.0),
                length_mm=bounds.width,
                rotation_z_deg=0.0,
            )
        if anchor == "north":
            return cls(
                anchor=anchor,
                origin=PointMM(bounds.min_x, bounds.max_y, bounds.min_z),
                tangent=(1.0, 0.0),
                inward_normal=(0.0, -1.0),
                length_mm=bounds.width,
                rotation_z_deg=180.0,
            )
        if anchor == "east":
            return cls(
                anchor=anchor,
                origin=PointMM(bounds.max_x, bounds.min_y, bounds.min_z),
                tangent=(0.0, 1.0),
                inward_normal=(-1.0, 0.0),
                length_mm=bounds.depth,
                rotation_z_deg=90.0,
            )
        return cls(
            anchor=anchor,
            origin=PointMM(bounds.min_x, bounds.min_y, bounds.min_z),
            tangent=(0.0, 1.0),
            inward_normal=(1.0, 0.0),
            length_mm=bounds.depth,
            rotation_z_deg=270.0,
        )

    def center_for(
        self,
        *,
        offset_mm: float,
        depth_mm: float,
        height_mm: float,
        z_center_mm: float | None = None,
    ) -> PointMM:
        z = z_center_mm if z_center_mm is not None else height_mm / 2.0
        return PointMM(
            x=(
                self.origin.x
                + (self.tangent[0] * offset_mm)
                + (self.inward_normal[0] * depth_mm / 2.0)
            ),
            y=(
                self.origin.y
                + (self.tangent[1] * offset_mm)
                + (self.inward_normal[1] * depth_mm / 2.0)
            ),
            z=z,
        )

    def span_for_center(self, *, center_offset_mm: float, width_mm: float) -> tuple[float, float]:
        half_width = width_mm / 2.0
        return (center_offset_mm - half_width, center_offset_mm + half_width)

    def is_offset_on_wall(self, offset_mm: float) -> bool:
        return 0.0 <= offset_mm <= self.length_mm


def normalize_rotation_z(rotation_z_deg: float) -> float:
    normalized = rotation_z_deg % 360.0
    if math.isclose(normalized, 360.0):
        return 0.0
    return normalized


def rotation_for_anchor(anchor: WallAnchor) -> float:
    return WallCoordinateSystem.from_bounds(
        anchor,
        RoomBounds(min_x=0.0, min_y=0.0, min_z=0.0, max_x=1.0, max_y=1.0, max_z=1.0),
    ).rotation_z_deg


def aabb_for_item(item: LayoutItem, dimensions: SchemaDimensionsMM | None = None) -> AABB:
    if dimensions is None:
        if item.dimensions_mm is None:
            raise ValueError("Cannot compute AABB for an item without dimensions.")
        dimensions = item.dimensions_mm
    return AABB.from_center(
        center=PointMM.from_schema(item.position_mm),
        dimensions=DimensionsMM.from_schema(dimensions),
        rotation_z_deg=item.rotation_z_deg,
    )
