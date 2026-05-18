"""Catalog loading, indexing, and SKU validation."""

from auto_design.catalog.colors import (
    CANONICAL_COLOR_KEYWORDS,
    ColorResolution,
    delta_e_cie76,
    resolve_color_term,
    resolve_intent_colors,
    resolve_prompt_colors,
)
from auto_design.catalog.service import (
    CatalogError,
    CatalogService,
    DuplicateSkuError,
    UnknownSkuError,
)

__all__ = [
    "CatalogError",
    "CatalogService",
    "ColorResolution",
    "DuplicateSkuError",
    "UnknownSkuError",
    "CANONICAL_COLOR_KEYWORDS",
    "delta_e_cie76",
    "resolve_color_term",
    "resolve_intent_colors",
    "resolve_prompt_colors",
]
