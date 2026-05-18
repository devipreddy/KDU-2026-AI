from __future__ import annotations

import json
from pathlib import Path

from auto_design.agent import run_planning_graph


ROOT = Path(__file__).resolve().parents[2]


def load_json(name: str) -> dict[str, object]:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _trace_by_rule(variant: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(entry["rule_id"]): entry
        for entry in variant["explainability_trace"]
    }


def test_explainability_trace_records_ranked_layout_decisions() -> None:
    result = run_planning_graph({"raw_input": load_json("input2.json")})
    variant = result["variants"][0]
    trace = variant["explainability_trace"]
    by_rule = _trace_by_rule(variant)

    assert variant["rationale"] == [
        {"rule_id": entry["rule_id"], "text": entry["text"]}
        for entry in trace
    ]
    assert {
        "topology",
        "retrieval",
        "workflow",
        "repair",
        "constraint",
        "scoring",
    } <= {entry["category"] for entry in trace}
    assert by_rule["TOPOLOGY-01"]["status"] == "pass"
    assert "L-shaped" in by_rule["TOPOLOGY-01"]["text"]
    assert "SKU-C11" in by_rule["COLOR-01"]["text"]
    assert by_rule["WORKFLOW-01"]["status"] == "pass"
    assert by_rule["LAYOUT-06"]["status"] == "pass"
    assert any(entry["category"] == "repair" for entry in trace)
    assert by_rule["TRADEOFF-01"]["status"] == "tradeoff"
    assert by_rule["SCORE-01"]["evidence"]["score"] == variant["score"]


def test_explainability_records_catalog_default_color_behavior() -> None:
    payload = load_json("input1.json")
    assert isinstance(payload["preferences"], dict)
    payload["preferences"]["prompt"] = "any modern style is fine"

    result = run_planning_graph({"raw_input": payload})
    by_rule = _trace_by_rule(result["variants"][0])

    assert by_rule["COLOR-DEFAULT"]["status"] == "pass"
    assert "catalog default SKUs" in by_rule["COLOR-DEFAULT"]["text"]
    assert by_rule["LAYOUT-01"]["status"] == "not_applicable"


def test_explainability_records_window_and_workflow_evidence() -> None:
    result = run_planning_graph({"raw_input": load_json("input3.json")})
    by_rule = _trace_by_rule(result["variants"][0])

    assert by_rule["LAYOUT-01"]["status"] == "pass"
    assert by_rule["LAYOUT-01"]["evidence"]["window_id"] == "east_window"
    assert by_rule["LAYOUT-01"]["evidence"]["delta_mm"] <= 300
    assert by_rule["WORKFLOW-01"]["evidence"]["gap_mm"] <= 600
