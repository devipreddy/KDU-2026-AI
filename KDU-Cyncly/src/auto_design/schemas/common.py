from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class ContractModel(BaseModel):
    """Base model for strict JSON contracts exchanged inside the planner."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


HexColor = Annotated[str, StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$")]
NonNegativeMM = Annotated[float, Field(ge=0)]
PositiveMM = Annotated[float, Field(gt=0)]


class Point3D(ContractModel):
    x: float
    y: float
    z: float


class DimensionsMM(ContractModel):
    width: PositiveMM
    depth: PositiveMM
    height: PositiveMM


class WallDimensions(ContractModel):
    height: PositiveMM
    length_mm: PositiveMM | None = None
