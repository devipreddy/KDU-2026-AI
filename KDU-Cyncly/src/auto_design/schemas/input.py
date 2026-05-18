from __future__ import annotations

from typing import Literal

from pydantic import Field

from auto_design.schemas.common import ContractModel
from auto_design.schemas.environment import Environment


BudgetTier = Literal["low", "mid", "high"]


class Preferences(ContractModel):
    budget_tier: BudgetTier
    must_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    prompt: str = ""
    catalog: str | None = None
    catalog_id: str | None = Field(default=None, alias="catalogId")


class DesignInput(ContractModel):
    environment: Environment
    preferences: Preferences
