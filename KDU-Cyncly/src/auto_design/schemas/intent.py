from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from auto_design.schemas.common import ContractModel, HexColor


LayoutFamilyCode = Literal["L", "U", "I"]
IntentTarget = Literal[
    "layout",
    "cabinets",
    "base_cabinets",
    "wall_cabinets",
    "tall_cabinets",
    "appliances",
    "sinks",
    "fixtures",
    "storage",
    "workflow",
    "general",
]
PromptConstraintKind = Literal[
    "topology",
    "style",
    "color",
    "material",
    "item_required",
    "item_excluded",
    "cabinet_scope",
    "storage",
    "workflow",
    "accessibility",
    "other",
]


class ColorRequest(ContractModel):
    raw_text: str = Field(min_length=1)
    target: IntentTarget = "cabinets"
    requested_hex: HexColor | None = None
    resolved_hex: HexColor | None = None
    matched_skus: list[str] = Field(default_factory=list)


class MaterialRequest(ContractModel):
    raw_text: str = Field(min_length=1)
    target: IntentTarget = "general"
    material: str = Field(min_length=1)


class PromptConstraint(ContractModel):
    kind: PromptConstraintKind
    text: str = Field(min_length=1)
    target: IntentTarget = "general"


class StructuredIntent(ContractModel):
    source_prompt: str | None = None
    layout_family: LayoutFamilyCode | None = None
    style: str | None = None
    style_tags: list[str] = Field(default_factory=list)
    color_requests: list[ColorRequest] = Field(default_factory=list)
    cabinet_color: ColorRequest | None = None
    material_requests: list[MaterialRequest] = Field(default_factory=list)
    required_items: list[str] = Field(default_factory=list)
    excluded_items: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    upper_cabinets: bool | None = None
    base_cabinets_only: bool | None = None
    pantry_storage: bool | None = None
    tall_cabinets: bool | None = None
    prompt_constraints: list[PromptConstraint] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def keep_legacy_item_lists_in_sync(self) -> StructuredIntent:
        if not self.required_items and self.must_have:
            self.required_items = list(self.must_have)
        if not self.must_have and self.required_items:
            self.must_have = list(self.required_items)
        if not self.excluded_items and self.avoid:
            self.excluded_items = list(self.avoid)
        if not self.avoid and self.excluded_items:
            self.avoid = list(self.excluded_items)
        if self.cabinet_color and self.cabinet_color not in self.color_requests:
            self.color_requests = [self.cabinet_color, *self.color_requests]
        return self
