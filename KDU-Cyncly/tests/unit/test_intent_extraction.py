from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from auto_design.agent.intent import (
    INTENT_SYSTEM_PROMPT,
    StructuredIntentExtractor,
    build_intent_messages,
    build_intent_payload,
)
from auto_design.schemas import DesignInput, StructuredIntent


ROOT = Path(__file__).resolve().parents[2]


def load_design_input(name: str = "input2.json") -> DesignInput:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return DesignInput.model_validate(payload)


def test_structured_intent_captures_semantic_preferences_without_geometry() -> None:
    intent = StructuredIntent.model_validate(
        {
            "source_prompt": "L-shape with navy base cabinets, no uppers, pantry storage",
            "layout_family": "L",
            "style": "modern",
            "style_tags": ["modern"],
            "color_requests": [
                {
                    "raw_text": "navy base cabinets",
                    "target": "base_cabinets",
                    "requested_hex": "#1F3A5F",
                }
            ],
            "material_requests": [
                {
                    "raw_text": "walnut shelves",
                    "target": "wall_cabinets",
                    "material": "walnut",
                }
            ],
            "required_items": ["dishwasher", "hood"],
            "excluded_items": ["double_sink"],
            "upper_cabinets": False,
            "base_cabinets_only": True,
            "pantry_storage": True,
            "tall_cabinets": True,
            "prompt_constraints": [
                {
                    "kind": "cabinet_scope",
                    "target": "wall_cabinets",
                    "text": "No upper cabinets.",
                }
            ],
        }
    )

    assert intent.layout_family == "L"
    assert intent.color_requests[0].target == "base_cabinets"
    assert intent.material_requests[0].material == "walnut"
    assert intent.must_have == ["dishwasher", "hood"]
    assert intent.avoid == ["double_sink"]
    assert intent.upper_cabinets is False
    assert intent.pantry_storage is True


def test_structured_intent_rejects_geometry_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        StructuredIntent.model_validate(
            {
                "layout_family": "L",
                "position_mm": {"x": 1200, "y": 3000, "z": 450},
            }
        )


def test_intent_prompt_explicitly_forbids_coordinate_generation() -> None:
    lowered = INTENT_SYSTEM_PROMPT.casefold()

    assert "never output geometry" in lowered
    assert "coordinates" in lowered
    assert "deterministic python" in lowered


def test_build_intent_payload_uses_preferences_without_environment_coordinates() -> None:
    design_input = load_design_input()

    payload = build_intent_payload(design_input)

    assert payload == {
        "prompt": "I want navy blue base cabinets",
        "budget_tier": "high",
        "must_have": ["dishwasher", "hood"],
        "avoid": [],
    }


def test_build_intent_messages_wraps_prompt_for_structured_llm() -> None:
    messages = build_intent_messages("only base cabinets, no uppers")

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "StructuredIntent" in messages[0].content
    assert "Do not infer or invent coordinates" in messages[1].content
    assert "only base cabinets, no uppers" in messages[1].content


class FakeStructuredRunnable:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.messages: list[Any] | None = None

    def invoke(self, input: list[Any]) -> dict[str, Any]:
        self.messages = input
        return self.result


class FakeStructuredOutputModel:
    def __init__(self, result: dict[str, Any]) -> None:
        self.schema: type[StructuredIntent] | None = None
        self.runnable = FakeStructuredRunnable(result)

    def with_structured_output(self, schema: type[StructuredIntent]) -> FakeStructuredRunnable:
        self.schema = schema
        return self.runnable


def test_intent_extractor_requires_structured_output_schema() -> None:
    fake_llm = FakeStructuredOutputModel(
        {
            "layout_family": "U",
            "style": "transitional",
            "required_items": ["dishwasher"],
            "upper_cabinets": True,
        }
    )

    intent = StructuredIntentExtractor(fake_llm).extract("U-shaped kitchen with dishwasher")

    assert fake_llm.schema is StructuredIntent
    assert intent.layout_family == "U"
    assert intent.must_have == ["dishwasher"]
    assert fake_llm.runnable.messages is not None
