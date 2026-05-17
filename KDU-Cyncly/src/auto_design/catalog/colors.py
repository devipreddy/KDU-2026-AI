from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from auto_design.catalog.service import CatalogService
from auto_design.schemas.catalog import Product
from auto_design.schemas.intent import ColorRequest, StructuredIntent


ColorResolutionStatus = Literal["resolved", "unsupported", "no_request"]

HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")
DEFAULT_DELTA_E_TOLERANCE = 8.0

CANONICAL_COLOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "oak": ("#D4A574",),
    "walnut": ("#6F4E37",),
    "matte white": ("#EDEDE9",),
    "matte black": ("#2E2E2E",),
    "navy blue": ("#1F3A5F",),
    "navy": ("#1F3A5F",),
    "cream": ("#F5E6CA",),
    "forest green": ("#2F5233",),
    "soft gray": ("#B0B8BE",),
    "sage green": ("#9CAF88",),
    "terracotta": ("#C76A4A",),
    "maple": ("#C8A878",),
    "shaker white": ("#E8E2D5",),
    "espresso": ("#4A3328",),
    "birch": ("#DBC59A",),
    "charcoal": ("#3D3D3D",),
    "stainless": ("#BFC1C2", "#C5C7C8"),
    "graphite": ("#2A2A2A", "#3A3A3A"),
    "brushed steel": ("#B8BABC",),
    "chrome": ("#D6D8DA",),
    "composite black": ("#2F2F2F",),
}

UNSUPPORTED_COLOR_WORDS = (
    "lavender",
    "pink",
    "purple",
    "yellow",
    "gold",
    "copper",
    "red",
    "orange",
    "teal",
)


@dataclass(frozen=True)
class ColorResolution:
    raw_text: str
    target: str
    status: ColorResolutionStatus
    requested_hex: str | None = None
    resolved_hex: str | None = None
    matched_skus: tuple[str, ...] = ()
    delta_e: float | None = None
    reason: str | None = None

    def to_color_request(self) -> ColorRequest:
        return ColorRequest(
            raw_text=self.raw_text,
            target=self.target,
            requested_hex=self.requested_hex,
            resolved_hex=self.resolved_hex,
            matched_skus=list(self.matched_skus),
        )


def normalize_hex(value: str) -> str:
    if not HEX_RE.fullmatch(value):
        raise ValueError(f"Invalid hex color: {value}")
    return value.upper()


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"\b{re.escape(phrase)}s?\b"
    return re.search(pattern, text) is not None


def infer_color_target(prompt: str, term: str) -> str:
    lowered = prompt.casefold()
    term_position = lowered.find(term.casefold())
    if term_position < 0:
        return "cabinets"
    before = lowered[:term_position]
    after = lowered[term_position + len(term) :]
    context = f"{before[-50:]} {after[:70]}"
    if "base cabinet" in context or "base cabinets" in context:
        return "base_cabinets"
    if "upper" in context or "wall cabinet" in context or "wall cabinets" in context:
        return "wall_cabinets"
    if "tall" in context or "pantry" in context:
        return "tall_cabinets"
    if "appliance" in context or "fridge" in context or "stainless" in term:
        return "appliances"
    if "sink" in context:
        return "sinks"
    if "fixture" in context or "tap" in context or "chrome" in term:
        return "fixtures"
    return "cabinets"


def _target_matches(product: Product, target: str) -> bool:
    if target == "base_cabinets":
        return product.category == "cabinet" and (
            product.type.startswith("base_") or "base_" in product.type
        )
    if target == "wall_cabinets":
        return product.category == "cabinet" and product.type.startswith("wall_")
    if target == "tall_cabinets":
        return product.category == "cabinet" and product.type.startswith("tall_")
    if target == "cabinets":
        return product.category == "cabinet"
    if target == "appliances":
        return product.category == "appliance"
    if target == "sinks":
        return product.type.startswith("sink_")
    if target == "fixtures":
        return product.category == "fixture" and not product.type.startswith("sink_")
    return True


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    normalized = normalize_hex(hex_color).lstrip("#")
    return tuple(int(normalized[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _pivot_rgb(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _pivot_xyz(value: float) -> float:
    if value > 0.008856:
        return value ** (1 / 3)
    return (7.787 * value) + (16 / 116)


def hex_to_lab(hex_color: str) -> tuple[float, float, float]:
    red, green, blue = (_pivot_rgb(channel) for channel in _hex_to_rgb01(hex_color))
    x = (red * 0.4124 + green * 0.3576 + blue * 0.1805) / 0.95047
    y = red * 0.2126 + green * 0.7152 + blue * 0.0722
    z = (red * 0.0193 + green * 0.1192 + blue * 0.9505) / 1.08883
    fx, fy, fz = (_pivot_xyz(channel) for channel in (x, y, z))
    return ((116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz))


def delta_e_cie76(first_hex: str, second_hex: str) -> float:
    first = hex_to_lab(first_hex)
    second = hex_to_lab(second_hex)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))


def _catalog_candidates(catalog: CatalogService, target: str) -> tuple[Product, ...]:
    return tuple(product for product in catalog.products if _target_matches(product, target))


def _nearest_matches(
    catalog: CatalogService,
    requested_hexes: tuple[str, ...],
    *,
    target: str,
    tolerance: float,
) -> tuple[str | None, tuple[str, ...], float | None]:
    candidates = _catalog_candidates(catalog, target)
    best_hex: str | None = None
    best_distance: float | None = None
    best_skus: list[str] = []

    for requested_hex in requested_hexes:
        normalized_requested = normalize_hex(requested_hex)
        distances = [
            (delta_e_cie76(normalized_requested, product.color), product)
            for product in candidates
        ]
        if not distances:
            continue
        candidate_distance = min(distance for distance, _product in distances)
        if candidate_distance > tolerance:
            continue
        matched_products = [
            product
            for distance, product in distances
            if abs(distance - candidate_distance) <= 0.001
        ]
        if best_distance is None or candidate_distance < best_distance:
            best_hex = normalize_hex(matched_products[0].color)
            best_distance = candidate_distance
            best_skus = [product.id for product in matched_products]
        elif abs(candidate_distance - best_distance) <= 0.001:
            best_skus.extend(product.id for product in matched_products)

    return best_hex, tuple(dict.fromkeys(best_skus)), best_distance


def resolve_color_term(
    catalog: CatalogService,
    raw_text: str,
    *,
    target: str = "cabinets",
    tolerance: float = DEFAULT_DELTA_E_TOLERANCE,
) -> ColorResolution:
    stripped = raw_text.strip()
    if not stripped:
        return ColorResolution(raw_text=raw_text, target=target, status="no_request")

    hex_match = HEX_RE.search(stripped)
    if hex_match:
        requested_hex = normalize_hex(hex_match.group(0))
        exact_matches = [
            product.id
            for product in _catalog_candidates(catalog, target)
            if product.color.upper() == requested_hex
        ]
        if exact_matches:
            return ColorResolution(
                raw_text=raw_text,
                target=target,
                status="resolved",
                requested_hex=requested_hex,
                resolved_hex=requested_hex,
                matched_skus=tuple(exact_matches),
                delta_e=0.0,
            )
        return ColorResolution(
            raw_text=raw_text,
            target=target,
            status="unsupported",
            requested_hex=requested_hex,
            reason="Hex literal did not exactly match catalog colors for the target.",
        )

    lowered = stripped.casefold()
    for keyword in sorted(CANONICAL_COLOR_KEYWORDS, key=len, reverse=True):
        if not _contains_phrase(lowered, keyword):
            continue
        requested_hexes = tuple(normalize_hex(value) for value in CANONICAL_COLOR_KEYWORDS[keyword])
        resolved_hex, matched_skus, distance = _nearest_matches(
            catalog,
            requested_hexes,
            target=target,
            tolerance=tolerance,
        )
        if matched_skus and resolved_hex:
            return ColorResolution(
                raw_text=raw_text,
                target=target,
                status="resolved",
                requested_hex=requested_hexes[0],
                resolved_hex=resolved_hex,
                matched_skus=matched_skus,
                delta_e=distance,
            )
        return ColorResolution(
            raw_text=raw_text,
            target=target,
            status="unsupported",
            requested_hex=requested_hexes[0],
            reason=f"No catalog colors matched '{keyword}' within delta E tolerance.",
        )

    for unsupported in UNSUPPORTED_COLOR_WORDS:
        if _contains_phrase(lowered, unsupported):
            return ColorResolution(
                raw_text=raw_text,
                target=target,
                status="unsupported",
                reason=f"Unsupported color keyword: {unsupported}.",
            )

    return ColorResolution(raw_text=raw_text, target=target, status="no_request")


def resolve_prompt_colors(
    catalog: CatalogService,
    prompt: str,
    *,
    tolerance: float = DEFAULT_DELTA_E_TOLERANCE,
) -> tuple[ColorResolution, ...]:
    lowered = prompt.casefold()
    resolutions: list[ColorResolution] = []
    seen_terms: set[str] = set()

    for match in HEX_RE.finditer(prompt):
        term = match.group(0)
        target = infer_color_target(prompt, term)
        resolution = resolve_color_term(catalog, term, target=target, tolerance=tolerance)
        resolutions.append(resolution)
        seen_terms.add(term.casefold())

    for keyword in sorted(CANONICAL_COLOR_KEYWORDS, key=len, reverse=True):
        if keyword in seen_terms:
            continue
        if keyword == "navy" and _contains_phrase(lowered, "navy blue"):
            continue
        if not _contains_phrase(lowered, keyword):
            continue
        target = infer_color_target(prompt, keyword)
        resolution = resolve_color_term(catalog, keyword, target=target, tolerance=tolerance)
        resolutions.append(resolution)
        seen_terms.add(keyword)

    if resolutions:
        return tuple(resolutions)

    for unsupported in UNSUPPORTED_COLOR_WORDS:
        if _contains_phrase(lowered, unsupported):
            target = infer_color_target(prompt, unsupported)
            return (
                resolve_color_term(catalog, unsupported, target=target, tolerance=tolerance),
            )

    return ()


def resolve_intent_colors(
    catalog: CatalogService,
    intent: StructuredIntent,
    *,
    tolerance: float = DEFAULT_DELTA_E_TOLERANCE,
) -> StructuredIntent:
    if intent.color_requests:
        resolved_requests = [
            resolve_color_term(
                catalog,
                request.requested_hex or request.raw_text,
                target=request.target,
                tolerance=tolerance,
            ).to_color_request()
            for request in intent.color_requests
        ]
    elif intent.source_prompt:
        resolved_requests = [
            resolution.to_color_request()
            for resolution in resolve_prompt_colors(
                catalog,
                intent.source_prompt,
                tolerance=tolerance,
            )
            if resolution.status == "resolved"
        ]
    else:
        resolved_requests = []

    if not resolved_requests:
        return intent.model_copy(update={"color_requests": [], "cabinet_color": None}, deep=True)

    cabinet_color = next(
        (
            request
            for request in resolved_requests
            if request.target in {"cabinets", "base_cabinets", "wall_cabinets", "tall_cabinets"}
        ),
        resolved_requests[0],
    )
    return intent.model_copy(
        update={
            "color_requests": resolved_requests,
            "cabinet_color": cabinet_color,
        },
        deep=True,
    )
