from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from auto_design.agent.fallback import DeterministicIntentParser
from auto_design.schemas.input import DesignInput
from auto_design.schemas.intent import StructuredIntent


INTENT_SYSTEM_PROMPT = """\
You extract kitchen-design intent for a deterministic spatial planner.

Return only data that matches the StructuredIntent schema. Extract layout family,
style, required items, exclusions, colors, materials, cabinet preferences, pantry
or tall-cabinet needs, and natural-language constraints.

Never output geometry. Do not include coordinates, wall positions, dimensions,
bounding boxes, rotations, or generated layout objects. The deterministic Python
geometry engine owns every coordinate and placement decision.
"""


class StructuredRunnable(Protocol):
    def invoke(self, input: Sequence[Any]) -> StructuredIntent | dict[str, Any]:
        """Invoke a structured-output model."""


class StructuredOutputModel(Protocol):
    def with_structured_output(self, schema: type[StructuredIntent]) -> StructuredRunnable:
        """Bind the model to a Pydantic structured-output schema."""


def build_intent_payload(request: DesignInput | str) -> dict[str, object]:
    if isinstance(request, str):
        return {
            "prompt": request,
            "budget_tier": None,
            "must_have": [],
            "avoid": [],
        }

    return {
        "prompt": request.preferences.prompt,
        "budget_tier": request.preferences.budget_tier,
        "must_have": request.preferences.must_have,
        "avoid": request.preferences.avoid,
    }


def build_intent_messages(request: DesignInput | str) -> list[SystemMessage | HumanMessage]:
    payload = build_intent_payload(request)
    return [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Extract structured semantic intent from this kitchen request. "
                "Do not infer or invent coordinates.\n\n"
                f"{json.dumps(payload, indent=2)}"
            )
        ),
    ]


class StructuredIntentExtractor:
    """LLM adapter that requires Pydantic structured output for semantic intent."""

    def __init__(self, llm: StructuredOutputModel) -> None:
        self._structured_llm = llm.with_structured_output(StructuredIntent)

    def extract(self, request: DesignInput | str) -> StructuredIntent:
        result = self._structured_llm.invoke(build_intent_messages(request))
        if isinstance(result, StructuredIntent):
            return result
        return StructuredIntent.model_validate(result)


def extract_intent(
    request: DesignInput | str,
    llm: StructuredOutputModel | None = None,
) -> StructuredIntent:
    if llm is None:
        return DeterministicIntentParser().parse(request)
    return StructuredIntentExtractor(llm).extract(request)
