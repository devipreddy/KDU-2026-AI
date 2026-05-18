from __future__ import annotations

import json
import uuid
from pathlib import Path

from auto_design.agent import run_planning_graph
from auto_design.schemas import LayoutResponse


ROOT = Path(__file__).resolve().parents[2]


def load_json(name: str) -> dict[str, object]:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_planning_graph_assembles_renderer_ready_output_envelope() -> None:
    result = run_planning_graph({"raw_input": load_json("input3.json")})
    output = result["output"]

    assert set(output) == {"request_id", "duration_ms", "layouts"}
    uuid.UUID(output["request_id"])
    assert 0 <= output["duration_ms"] < 30_000
    assert len(output["layouts"]) == 5
    LayoutResponse.model_validate(output)

    first_layout = output["layouts"][0]
    assert set(first_layout) == {
        "id",
        "family",
        "score",
        "violations",
        "environment",
        "layout",
        "rationale",
    }
    assert first_layout["family"] == "L-shaped"
    assert first_layout["environment"]["openings"]
    assert first_layout["layout"]["north_wall"]["is_wall"] is True
    assert first_layout["layout"]["south_door"]["is_door"] is True
    assert first_layout["layout"]["north_window"]["is_window"] is True
    assert first_layout["layout"]["east_window"]["is_window"] is True
    assert first_layout["rationale"]
    assert all(
        set(rationale) == {"rule_id", "text"}
        for rationale in first_layout["rationale"]
    )


def test_output_envelope_is_deterministic_for_same_request() -> None:
    payload = load_json("input2.json")

    first = run_planning_graph({"raw_input": payload})["output"]
    second = run_planning_graph({"raw_input": payload})["output"]

    assert first == second
    assert first["layouts"][0]["layout"]["north_wall"]["is_wall"] is True
    assert any(
        item.get("product_id")
        for item in first["layouts"][0]["layout"].values()
    )
