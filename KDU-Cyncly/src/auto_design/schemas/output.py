from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from auto_design.schemas.common import ContractModel, DimensionsMM, Point3D
from auto_design.schemas.environment import Environment, WallAnchor


ZoneType = Literal["cooking", "cleaning", "cooling", "preparation", "storage", "default"]
ViolationSeverity = Literal["hard", "soft", "warning", "error"]


class LayoutItem(ContractModel):
    position_mm: Point3D
    rotation_z_deg: float
    dimensions_mm: DimensionsMM | None = None
    product_id: str | None = None
    anchor_wall: WallAnchor | None = None
    zone_type: ZoneType | None = None
    is_wall: bool | None = None
    is_door: bool | None = None
    is_window: bool | None = None

    @model_validator(mode="after")
    def validate_item_identity(self) -> LayoutItem:
        is_structure = bool(self.is_wall or self.is_door or self.is_window)
        if not self.product_id and not is_structure:
            raise ValueError("layout items must be either structural or reference a product_id")
        if is_structure and self.dimensions_mm is None:
            raise ValueError("structural layout items must include dimensions_mm")
        return self


class Violation(ContractModel):
    rule_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    severity: ViolationSeverity | None = None


class Rationale(ContractModel):
    rule_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class LayoutVariant(ContractModel):
    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    violations: list[Violation] = Field(default_factory=list)
    environment: Environment
    layout: dict[str, LayoutItem] = Field(min_length=1)
    rationale: list[Rationale] = Field(default_factory=list)


class LayoutResponse(ContractModel):
    request_id: UUID
    duration_ms: int = Field(ge=0)
    layouts: list[LayoutVariant] = Field(min_length=1, max_length=5)
