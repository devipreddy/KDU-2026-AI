from __future__ import annotations

from typing import Literal

from pydantic import Field, RootModel

from auto_design.schemas.common import ContractModel, HexColor, NonNegativeMM, PositiveMM


ProductCategory = Literal["cabinet", "appliance", "fixture", "sink", "door", "window"]
AttachmentMode = Literal["wall", "floor", "ceiling", "corner", "none"]
PriceTier = Literal["low", "mid", "high"]


class ProductConstraints(ContractModel):
    front_clearance_mm: NonNegativeMM | None = None
    needs_water: bool = False
    needs_power: bool = False


class Product(ContractModel):
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    category: ProductCategory
    color: HexColor
    width_mm: PositiveMM
    depth_mm: PositiveMM
    height_mm: PositiveMM
    must_attach_to: AttachmentMode
    style_tags: list[str] = Field(default_factory=list)
    price_tier: PriceTier
    constraints: ProductConstraints = Field(default_factory=ProductConstraints)


class ProductCatalog(RootModel[list[Product]]):
    root: list[Product] = Field(min_length=1)
