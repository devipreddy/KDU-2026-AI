from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PricingRate:
    input_per_million: float
    output_per_million: float = 0.0


MODEL_PRICING: dict[str, PricingRate] = {
    "gpt-4o-mini": PricingRate(input_per_million=0.15, output_per_million=0.60),
    "openai/gpt-4o-mini": PricingRate(input_per_million=0.15, output_per_million=0.60),
    "text-embedding-3-small": PricingRate(input_per_million=0.02, output_per_million=0.0),
    "openai/text-embedding-3-small": PricingRate(input_per_million=0.02, output_per_million=0.0),
}

VISION_OPERATIONS = {"vision_pdf_page", "vision_image_extract", "vision_pdf_visuals"}
ENRICHMENT_OPERATIONS = {"document_enrichment", "enrichment_chunk", "enrichment_merge"}
EMBEDDING_OPERATIONS = {"chunk_embeddings", "query_embedding"}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rate = MODEL_PRICING.get(model)
    if rate is None:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * rate.input_per_million
    output_cost = (output_tokens / 1_000_000) * rate.output_per_million
    return round(input_cost + output_cost, 8)


def classify_operation(operation: str) -> str:
    if operation in VISION_OPERATIONS:
        return "vision"
    if operation in ENRICHMENT_OPERATIONS:
        return "enrichment"
    if operation in EMBEDDING_OPERATIONS:
        return "embedding"
    return "other"
