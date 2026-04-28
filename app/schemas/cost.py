from __future__ import annotations

from app.schemas.common import APIModel


class CostOperationSummary(APIModel):
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class CostModelSummary(APIModel):
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class FileCostSummary(APIModel):
    file_id: str
    file_name: str
    vision_cost_usd: float
    enrichment_cost_usd: float
    embedding_cost_usd: float
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int


class CostDashboardResponse(APIModel):
    total_cost_usd: float
    files: list[FileCostSummary]
    by_operation: list[CostOperationSummary]
    by_model: list[CostModelSummary]
