from __future__ import annotations

import re
from dataclasses import dataclass

from auto_design.schemas.input import DesignInput
from auto_design.schemas.intent import (
    ColorRequest,
    LayoutFamilyCode,
    MaterialRequest,
    PromptConstraint,
    StructuredIntent,
)


STYLE_KEYWORDS = (
    "modern",
    "traditional",
    "minimalist",
    "transitional",
    "contemporary",
    "farmhouse",
    "industrial",
    "scandinavian",
)

COLOR_KEYWORDS = (
    "matte white",
    "matte black",
    "navy blue",
    "forest green",
    "soft gray",
    "sage green",
    "shaker white",
    "brushed steel",
    "composite black",
    "terracotta",
    "stainless",
    "graphite",
    "charcoal",
    "espresso",
    "walnut",
    "cream",
    "birch",
    "maple",
    "navy",
    "oak",
    "chrome",
)

MATERIAL_KEYWORDS = (
    "oak",
    "walnut",
    "maple",
    "birch",
    "stainless",
    "brushed steel",
    "chrome",
    "graphite",
)

APPLIANCE_KEYWORDS = {
    "dishwasher": ("dishwasher", "dish washer"),
    "hood": ("hood", "range hood", "extractor"),
    "fridge": ("fridge", "refrigerator"),
    "stove": ("stove", "cooktop", "range"),
    "oven": ("oven",),
    "microwave": ("microwave",),
    "single_sink": ("single sink",),
    "double_sink": ("double sink",),
    "sink": ("sink",),
}

NEGATION_PREFIXES = ("no", "without", "avoid", "exclude", "skip")


def _prompt_text(request: DesignInput | str) -> str:
    return request if isinstance(request, str) else request.preferences.prompt


def _preference_items(request: DesignInput | str) -> tuple[list[str], list[str]]:
    if isinstance(request, str):
        return [], []
    return list(request.preferences.must_have), list(request.preferences.avoid)


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"\b{re.escape(phrase)}s?\b"
    return re.search(pattern, text) is not None


def _is_negated(text: str, phrase: str) -> bool:
    phrase_pattern = re.escape(phrase)
    prefix = "|".join(NEGATION_PREFIXES)
    pattern = rf"\b(?:{prefix})\s+(?:a|an|the)?\s*{phrase_pattern}s?\b"
    return re.search(pattern, text) is not None


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _detect_layout_family(text: str) -> LayoutFamilyCode | None:
    if re.search(r"\bl[\s-]?shape(?:d)?\b", text):
        return "L"
    if re.search(r"\bu[\s-]?shape(?:d)?\b", text):
        return "U"
    if re.search(r"\bi[\s-]?shape(?:d)?\b", text):
        return "I"
    if re.search(r"\b(?:single|one)[\s-]?wall\b", text):
        return "I"
    if re.search(r"\blinear\b", text):
        return "I"
    return None


def _color_target(text: str, keyword: str) -> str:
    after_keyword = text[text.find(keyword) + len(keyword) :]
    before_keyword = text[: text.find(keyword)]
    context = f"{before_keyword[-40:]} {after_keyword[:60]}"
    if "base cabinet" in context or "base cabinets" in context:
        return "base_cabinets"
    if "upper" in context or "wall cabinet" in context or "wall cabinets" in context:
        return "wall_cabinets"
    if "tall" in context or "pantry" in context:
        return "tall_cabinets"
    if "appliance" in context or "fridge" in context or "stainless" in keyword:
        return "appliances"
    return "cabinets"


@dataclass(frozen=True)
class DeterministicIntentParser:
    """Small keyword parser used when no LLM is configured."""

    def parse(self, request: DesignInput | str) -> StructuredIntent:
        prompt = _prompt_text(request)
        lowered = prompt.casefold()
        required, excluded = _preference_items(request)
        constraints: list[PromptConstraint] = []

        layout_family = _detect_layout_family(lowered)
        if layout_family:
            constraints.append(
                PromptConstraint(
                    kind="topology",
                    target="layout",
                    text=f"Requested {layout_family}-family kitchen layout.",
                )
            )

        style_tags = [style for style in STYLE_KEYWORDS if _contains_phrase(lowered, style)]
        style = style_tags[0] if style_tags else None
        for style_tag in style_tags:
            constraints.append(
                PromptConstraint(kind="style", target="general", text=f"Style: {style_tag}.")
            )

        color_requests: list[ColorRequest] = []
        seen_color_targets: set[tuple[str, str]] = set()
        for keyword in COLOR_KEYWORDS:
            if keyword == "navy" and _contains_phrase(lowered, "navy blue"):
                continue
            if not _contains_phrase(lowered, keyword):
                continue
            target = _color_target(lowered, keyword)
            key = (keyword, target)
            if key in seen_color_targets:
                continue
            seen_color_targets.add(key)
            raw_text = f"{keyword} {target.replace('_', ' ')}"
            color_requests.append(ColorRequest(raw_text=raw_text, target=target))
            constraints.append(
                PromptConstraint(kind="color", target=target, text=f"Color request: {keyword}.")
            )

        material_requests: list[MaterialRequest] = []
        for keyword in MATERIAL_KEYWORDS:
            if not _contains_phrase(lowered, keyword):
                continue
            target = _color_target(lowered, keyword)
            material_requests.append(
                MaterialRequest(
                    raw_text=f"{keyword} {target.replace('_', ' ')}",
                    target=target,
                    material=keyword,
                )
            )
            constraints.append(
                PromptConstraint(
                    kind="material",
                    target=target,
                    text=f"Material request: {keyword}.",
                )
            )

        for item, phrases in APPLIANCE_KEYWORDS.items():
            if item == "sink" and (
                _contains_phrase(lowered, "single sink")
                or _contains_phrase(lowered, "double sink")
            ):
                continue
            for phrase in phrases:
                if not _contains_phrase(lowered, phrase):
                    continue
                if _is_negated(lowered, phrase):
                    _append_unique(excluded, item)
                    constraints.append(
                        PromptConstraint(
                            kind="item_excluded",
                            target="appliances",
                            text=f"Excluded item: {item}.",
                        )
                    )
                else:
                    _append_unique(required, item)
                    constraints.append(
                        PromptConstraint(
                            kind="item_required",
                            target="appliances",
                            text=f"Required item: {item}.",
                        )
                    )
                break

        upper_cabinets: bool | None = None
        base_cabinets_only: bool | None = None
        if re.search(r"\bonly\s+base\s+cabinets?\b", lowered):
            base_cabinets_only = True
            upper_cabinets = False
            constraints.append(
                PromptConstraint(
                    kind="cabinet_scope",
                    target="base_cabinets",
                    text="Only base cabinets requested.",
                )
            )
        if re.search(r"\bno\s+(?:upper|uppers|wall\s+cabinets?)\b", lowered):
            upper_cabinets = False
            constraints.append(
                PromptConstraint(
                    kind="cabinet_scope",
                    target="wall_cabinets",
                    text="No upper cabinets requested.",
                )
            )
        elif re.search(r"\b(?:upper|uppers|wall\s+cabinets?|upper\s+storage)\b", lowered):
            upper_cabinets = True
            constraints.append(
                PromptConstraint(
                    kind="cabinet_scope",
                    target="wall_cabinets",
                    text="Upper cabinet storage requested.",
                )
            )

        pantry_storage = None
        tall_cabinets = None
        if _contains_phrase(lowered, "pantry") or _contains_phrase(lowered, "pantry storage"):
            pantry_storage = True
            tall_cabinets = True
            constraints.append(
                PromptConstraint(kind="storage", target="storage", text="Pantry storage requested.")
            )
        elif _contains_phrase(lowered, "tall cabinet") or _contains_phrase(
            lowered, "tall cabinets"
        ):
            tall_cabinets = True
            constraints.append(
                PromptConstraint(
                    kind="storage",
                    target="tall_cabinets",
                    text="Tall cabinet requested.",
                )
            )

        cabinet_color = color_requests[0] if color_requests else None
        return StructuredIntent(
            source_prompt=prompt,
            layout_family=layout_family,
            style=style,
            style_tags=style_tags,
            color_requests=color_requests,
            cabinet_color=cabinet_color,
            material_requests=material_requests,
            required_items=required,
            excluded_items=excluded,
            upper_cabinets=upper_cabinets,
            base_cabinets_only=base_cabinets_only,
            pantry_storage=pantry_storage,
            tall_cabinets=tall_cabinets,
            prompt_constraints=constraints,
        )
