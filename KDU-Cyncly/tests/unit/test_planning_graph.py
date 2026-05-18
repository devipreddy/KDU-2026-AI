from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from auto_design.agent import GRAPH_NODE_ORDER, build_planning_graph, run_planning_graph
from auto_design.agent.graph import infer_retrieval_categories
from auto_design.schemas import LayoutResponse, StructuredIntent


ROOT = Path(__file__).resolve().parents[2]


def load_json(name: str) -> dict[str, object]:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_planning_graph_compiles_with_required_nodes() -> None:
    graph = build_planning_graph()
    graph_repr = graph.get_graph()
    node_names = set(graph_repr.nodes)

    for node_name in GRAPH_NODE_ORDER:
        assert node_name in node_names


def test_planning_graph_runs_required_pipeline_in_order() -> None:
    result = run_planning_graph({"raw_input": load_json("input2.json")})

    assert result["trace"] == list(GRAPH_NODE_ORDER)
    assert result["prompt"] == "I want navy blue base cabinets"
    assert result["intent"].cabinet_color is not None
    assert result["intent"].cabinet_color.matched_skus == ["SKU-C11"]
    assert result["retrieval_categories"] == ["cabinets", "appliances"]
    assert result["retrieval_results"]["cabinets"]["count"] == 1
    assert result["feasibility"]["status"] == "feasible"
    assert result["feasibility"]["selected_family"] == "L"
    assert result["feasibility"]["topology_fits"]["L"]["feasible"] is True
    assert len(result["variants"]) == 5
    assert result["variants"][0]["family"] == "L"
    assert result["variants"][0]["status"] == "placed_template"
    assert result["variants"][0]["score"] == result["scores"][0]["score"]
    assert result["variants"][0]["rank"] == 1
    assert result["variants"][0]["confidence"]["geometry"] >= 0
    assert result["scores"][0]["weights"] == {
        "geometry": 0.4,
        "workflow": 0.3,
        "retrieval_color": 0.2,
        "topology": 0.1,
    }
    assert [score["score"] for score in result["scores"]] == sorted(
        [score["score"] for score in result["scores"]],
        reverse=True,
    )
    assert result["variants"][0]["variant_seed"]
    assert result["variants"][0]["diversity"]["signature"]
    assert result["variants"][0]["topology"]["family"] == "L"
    assert result["variants"][0]["zone_plan"]["family"] == "L"
    assert result["variants"][0]["zone_plan"]["item_assignments"]
    assert result["variants"][0]["placement"]["is_continuous"] is True
    assert result["variants"][0]["placement"]["base_coverage_valid"] is True
    assert result["variants"][0]["placement"]["base_coverages"]
    assert result["variants"][0]["layout"]
    assert result["violations"] == []
    assert result["repairs"]
    assert all(
        {"variant_id", "rule_id", "action", "item_keys", "text"} <= set(repair)
        for repair in result["repairs"]
    )
    assert {
        repair["action"]
        for repair in result["repairs"]
    } >= {"compact_runs_to_shared_corner"}
    assert result["variants"][0]["violations"] == []
    assert result["variants"][0]["repair_history"]
    assert result["variants"][0]["rationale"]
    assert result["variants"][0]["explainability_trace"]
    assert {
        entry["rule_id"]
        for entry in result["variants"][0]["explainability_trace"]
    } >= {"TOPOLOGY-01", "COLOR-01", "WORKFLOW-01", "LAYOUT-06", "SCORE-01"}
    assert result["scores"][0]["variant_id"] == result["variants"][0]["id"]
    UUID(result["output"]["request_id"])
    assert result["output"]["duration_ms"] < 30_000
    assert len(result["output"]["layouts"]) == 5
    assert result["output"]["layouts"][0]["id"] == result["variants"][0]["id"]
    assert result["output"]["layouts"][0]["family"] == "L-shaped"
    assert result["output"]["layouts"][0]["score"] == result["variants"][0]["score"]
    assert result["output"]["layouts"][0]["rationale"] == result["variants"][0]["rationale"]
    LayoutResponse.model_validate(result["output"])


def test_graph_infers_retrieval_categories_from_intent_requirements() -> None:
    intent = StructuredIntent.model_validate(
        {
            "required_items": ["dishwasher", "hood", "single_sink"],
            "color_requests": [
                {
                    "raw_text": "chrome fixtures",
                    "target": "fixtures",
                    "requested_hex": "#D6D8DA",
                }
            ],
        }
    )

    assert infer_retrieval_categories(intent) == [
        "cabinets",
        "appliances",
        "sinks",
        "fixtures",
    ]


def test_planning_graph_preserves_existing_validated_input_objects() -> None:
    payload = load_json("input1.json")
    first_result = run_planning_graph({"raw_input": payload})
    second_result = run_planning_graph({"input": first_result["input"]})

    assert second_result["input"] == first_result["input"]
    assert second_result["trace"] == list(GRAPH_NODE_ORDER)


def test_planning_graph_returns_fallback_metadata_for_impossible_request() -> None:
    payload = load_json("input1.json")
    assert isinstance(payload["preferences"], dict)
    payload["preferences"]["prompt"] = "Give me a U-shaped kitchen with dishwasher and hood."

    result = run_planning_graph({"raw_input": payload})

    assert result["intent"].layout_family == "U"
    assert result["feasibility"]["status"] == "fallback"
    assert result["feasibility"]["requested_family"] == "U"
    assert result["feasibility"]["requested_family_feasible"] is False
    assert result["feasibility"]["selected_family"] == "L"
    assert {variant["family"] for variant in result["variants"]} == {"L"}
    assert {layout["family"] for layout in result["output"]["layouts"]} == {"L-shaped"}
    assert len(result["output"]["layouts"]) == len(result["variants"])


def test_planning_graph_requested_family_only_returns_that_family() -> None:
    payload = load_json("input1.json")
    assert isinstance(payload["preferences"], dict)
    payload["preferences"]["prompt"] = "single-wall kitchen with dishwasher and hood"

    result = run_planning_graph({"raw_input": payload})

    assert result["intent"].layout_family == "I"
    assert result["feasibility"]["selected_family"] == "I"
    assert {variant["family"] for variant in result["variants"]} == {"I"}
    assert {variant["topology"]["family"] for variant in result["variants"]} == {"I"}
    assert {variant["zone_plan"]["family"] for variant in result["variants"]} == {"I"}
    assert {variant["placement"]["family"] for variant in result["variants"]} == {"I"}
