from __future__ import annotations

import asyncio
import json
from pathlib import Path

from auto_design.catalog.service import CatalogService
from auto_design.planner import (
    analyze_feasibility,
    generate_layout_variants_async,
    variant_seed_for,
)
from auto_design.schemas import DesignInput, StructuredIntent


ROOT = Path(__file__).resolve().parents[2]


def load_design_input(name: str = "input2.json") -> DesignInput:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return DesignInput.model_validate(payload)


def load_catalog() -> CatalogService:
    return CatalogService.load(ROOT / "catalog.json")


def state_for(name: str = "input2.json"):
    design_input = load_design_input(name)
    catalog = load_catalog()
    intent = StructuredIntent(
        layout_family="L",
        required_items=design_input.preferences.must_have,
    )
    feasibility = analyze_feasibility(design_input, intent, catalog).to_payload()
    return design_input, intent, catalog, feasibility


def generate_once(name: str = "input2.json") -> list[dict[str, object]]:
    design_input, intent, catalog, feasibility = state_for(name)
    return asyncio.run(
        generate_layout_variants_async(
            design_input=design_input,
            intent=intent,
            feasibility=feasibility,
            catalog=catalog,
        )
    )


def test_async_variant_generation_returns_five_deterministic_variants() -> None:
    first = generate_once()
    second = generate_once()

    assert len(first) == 5
    assert [variant["id"] for variant in first] == [
        f"variant-l-{index}" for index in range(1, 6)
    ]
    assert [
        variant["diversity"]["signature"] for variant in first
    ] == [
        variant["diversity"]["signature"] for variant in second
    ]
    assert [variant["variant_seed"] for variant in first] == [
        variant["variant_seed"] for variant in second
    ]


def test_async_variants_differ_by_zone_wall_or_sequence_signature() -> None:
    variants = generate_once()
    signatures = {
        str(variant["diversity"]["signature"])
        for variant in variants
    }
    zone_maps = {
        json.dumps(variant["diversity"]["zone_wall_map"], sort_keys=True)
        for variant in variants
    }

    assert len(signatures) >= 3
    assert len(zone_maps) >= 3


def test_variant_seed_changes_with_prompt_text() -> None:
    design_input, intent, catalog, feasibility = state_for()
    original_seed = variant_seed_for(design_input, intent, feasibility)
    payload = design_input.model_dump(mode="json")
    assert isinstance(payload["preferences"], dict)
    payload["preferences"]["prompt"] = "single-wall kitchen with walnut cabinets"
    changed_input = DesignInput.model_validate(payload)
    changed_intent = StructuredIntent(
        layout_family="L",
        required_items=changed_input.preferences.must_have,
    )

    changed_seed = variant_seed_for(changed_input, changed_intent, feasibility)

    assert changed_seed != original_seed


def test_async_variants_respect_requested_i_family() -> None:
    design_input = load_design_input("input1.json")
    catalog = load_catalog()
    intent = StructuredIntent(
        layout_family="I",
        required_items=design_input.preferences.must_have,
    )
    feasibility = analyze_feasibility(design_input, intent, catalog).to_payload()

    variants = asyncio.run(
        generate_layout_variants_async(
            design_input=design_input,
            intent=intent,
            feasibility=feasibility,
            catalog=catalog,
        )
    )

    assert len(variants) == 5
    assert {variant["family"] for variant in variants} == {"I"}
    assert all(len(variant["topology"]["walls"]) == 1 for variant in variants)
