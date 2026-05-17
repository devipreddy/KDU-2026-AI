from __future__ import annotations

from pathlib import Path

from auto_design.agent import extract_intent
from auto_design.catalog.colors import (
    CANONICAL_COLOR_KEYWORDS,
    delta_e_cie76,
    resolve_intent_colors,
    resolve_prompt_colors,
)
from auto_design.catalog.service import CatalogService


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "catalog.json"


def catalog() -> CatalogService:
    return CatalogService.load(CATALOG_PATH)


def skus_for(prompt: str) -> list[str]:
    resolutions = resolve_prompt_colors(catalog(), prompt)
    assert len(resolutions) == 1
    return list(resolutions[0].matched_skus)


def test_canonical_keyword_map_contains_required_project_terms() -> None:
    assert CANONICAL_COLOR_KEYWORDS["navy"] == ("#1F3A5F",)
    assert CANONICAL_COLOR_KEYWORDS["navy blue"] == ("#1F3A5F",)
    assert CANONICAL_COLOR_KEYWORDS["walnut"] == ("#6F4E37",)
    assert CANONICAL_COLOR_KEYWORDS["terracotta"] == ("#C76A4A",)
    assert CANONICAL_COLOR_KEYWORDS["stainless"] == ("#BFC1C2", "#C5C7C8")
    assert CANONICAL_COLOR_KEYWORDS["graphite"] == ("#2A2A2A", "#3A3A3A")


def test_hex_literal_requires_exact_catalog_color_match() -> None:
    resolution = resolve_prompt_colors(catalog(), "Use #1F3A5F base cabinets")[0]

    assert resolution.status == "resolved"
    assert resolution.requested_hex == "#1F3A5F"
    assert resolution.resolved_hex == "#1F3A5F"
    assert resolution.matched_skus == ("SKU-C11",)
    assert resolution.delta_e == 0.0


def test_navy_blue_base_cabinets_resolve_to_catalog_sku() -> None:
    assert skus_for("I want navy blue base cabinets") == ["SKU-C11"]


def test_walnut_cabinets_resolve_to_catalog_sku() -> None:
    assert skus_for("walnut cabinets please") == ["SKU-C08"]


def test_terracotta_cabinets_resolve_to_catalog_sku() -> None:
    assert skus_for("terracotta cabinets") == ["SKU-C16"]


def test_stainless_appliances_resolve_to_stocked_appliance_skus() -> None:
    assert skus_for("stainless appliances") == ["SKU-A06", "SKU-A08", "SKU-A01", "SKU-A03"]


def test_graphite_appliances_resolve_to_stocked_appliance_skus() -> None:
    assert skus_for("graphite appliances") == ["SKU-A04", "SKU-A07"]


def test_no_color_prompt_returns_no_resolution_and_preserves_catalog_defaults() -> None:
    assert resolve_prompt_colors(catalog(), "any modern style is fine") == ()


def test_unsupported_color_returns_no_skus() -> None:
    resolution = resolve_prompt_colors(catalog(), "lavender cabinets")[0]

    assert resolution.status == "unsupported"
    assert resolution.resolved_hex is None
    assert resolution.matched_skus == ()


def test_delta_e_matching_respects_tolerance() -> None:
    assert delta_e_cie76("#1F3A5F", "#1F3A5F") == 0.0
    assert skus_for("navy cabinets") == ["SKU-C11"]


def test_resolve_intent_colors_updates_fallback_intent_with_skus() -> None:
    intent = extract_intent("L-shape with navy blue base cabinets")

    resolved = resolve_intent_colors(catalog(), intent)

    assert resolved.color_requests[0].resolved_hex == "#1F3A5F"
    assert resolved.color_requests[0].matched_skus == ["SKU-C11"]
    assert resolved.cabinet_color == resolved.color_requests[0]
