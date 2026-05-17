from __future__ import annotations

from typing import Literal

from pydantic import Field

from auto_design.schemas.common import ContractModel, HexColor


LayoutFamilyCode = Literal["L", "U", "I"]


class ColorRequest(ContractModel):
    raw_text: str = Field(min_length=1)
    resolved_hex: HexColor | None = None
    matched_skus: list[str] = Field(default_factory=list)


class StructuredIntent(ContractModel):
    layout_family: LayoutFamilyCode | None = None
    style: str | None = None
    cabinet_color: ColorRequest | None = None
    material_requests: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    upper_cabinets: bool | None = None
    pantry_storage: bool | None = None
    notes: list[str] = Field(default_factory=list)
