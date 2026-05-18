from __future__ import annotations

import json
from pathlib import Path

from auto_design.agent import extract_intent
from auto_design.agent.fallback import DeterministicIntentParser
from auto_design.schemas import DesignInput


ROOT = Path(__file__).resolve().parents[2]


def load_design_input(name: str = "input2.json") -> DesignInput:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return DesignInput.model_validate(payload)


def test_fallback_parser_detects_l_shape_and_navy_base_cabinets() -> None:
    intent = DeterministicIntentParser().parse(
        "Give me an L-shape kitchen with navy blue base cabinets."
    )

    assert intent.layout_family == "L"
    assert intent.color_requests
    assert intent.color_requests[0].raw_text == "navy blue base cabinets"
    assert intent.color_requests[0].target == "base_cabinets"
    assert intent.cabinet_color == intent.color_requests[0]
    assert {constraint.kind for constraint in intent.prompt_constraints} >= {
        "topology",
        "color",
    }


def test_fallback_parser_detects_no_uppers_and_only_base_cabinets() -> None:
    intent = DeterministicIntentParser().parse("only base cabinets, no uppers")

    assert intent.upper_cabinets is False
    assert intent.base_cabinets_only is True
    assert any(
        constraint.target == "wall_cabinets" and constraint.kind == "cabinet_scope"
        for constraint in intent.prompt_constraints
    )


def test_fallback_parser_detects_u_and_i_layout_phrases() -> None:
    parser = DeterministicIntentParser()

    assert parser.parse("U-shaped kitchen with lots of storage").layout_family == "U"
    assert parser.parse("single-wall kitchen with dishwasher").layout_family == "I"
    assert parser.parse("linear kitchen, any modern style is fine").layout_family == "I"


def test_fallback_parser_detects_pantry_and_tall_cabinet_need() -> None:
    intent = DeterministicIntentParser().parse("modern kitchen with pantry storage")

    assert intent.style == "modern"
    assert intent.pantry_storage is True
    assert intent.tall_cabinets is True
    assert any(constraint.kind == "storage" for constraint in intent.prompt_constraints)


def test_fallback_parser_detects_appliance_requests_and_exclusions() -> None:
    intent = DeterministicIntentParser().parse(
        "I need a dishwasher, hood, fridge, oven, microwave, and avoid double sink."
    )

    assert intent.must_have == ["dishwasher", "hood", "fridge", "oven", "microwave"]
    assert intent.avoid == ["double_sink"]
    assert "sink" not in intent.must_have


def test_fallback_parser_merges_input_preferences_with_prompt_keywords() -> None:
    design_input = load_design_input()

    intent = DeterministicIntentParser().parse(design_input)

    assert intent.source_prompt == "I want navy blue base cabinets"
    assert intent.must_have == ["dishwasher", "hood"]
    assert intent.color_requests[0].target == "base_cabinets"


def test_extract_intent_uses_fallback_parser_when_llm_is_absent() -> None:
    intent = extract_intent("L-shaped kitchen with dishwasher and no uppers")

    assert intent.layout_family == "L"
    assert intent.must_have == ["dishwasher"]
    assert intent.upper_cabinets is False
