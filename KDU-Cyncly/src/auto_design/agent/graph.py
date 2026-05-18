from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from auto_design.agent.intent import StructuredOutputModel, extract_intent
from auto_design.catalog.colors import resolve_intent_colors
from auto_design.catalog.retrieval import (
    AsyncCatalogRetriever,
    CatalogRetrievalQuery,
)
from auto_design.catalog.service import CatalogService
from auto_design.planner.feasibility import analyze_feasibility
from auto_design.planner.grammar import generate_topology_templates
from auto_design.planner.placement import generate_placement_plan
from auto_design.planner.zones import plan_zones_for_template
from auto_design.repair import (
    flatten_repair_actions,
    flatten_repair_violations,
    repair_variants,
)
from auto_design.schemas.input import DesignInput
from auto_design.schemas.intent import StructuredIntent
from auto_design.validation import flatten_validation_results, validate_variants


GRAPH_NODE_ORDER = (
    "validate_input",
    "extract_intent",
    "retrieve_catalog",
    "analyze_feasibility",
    "generate_variants",
    "validate",
    "repair",
    "score",
    "assemble_output",
)


class PlanningState(TypedDict, total=False):
    raw_input: dict[str, Any] | DesignInput
    input: DesignInput
    llm: StructuredOutputModel | None
    prompt: str
    catalog_path: str
    catalog: CatalogService
    intent: StructuredIntent
    retrieval_categories: list[str]
    retrieval_results: dict[str, dict[str, object]]
    feasibility: dict[str, object]
    variants: list[dict[str, object]]
    violations: list[dict[str, object]]
    repairs: list[dict[str, object]]
    scores: list[dict[str, object]]
    output: dict[str, object]
    trace: list[str]


def _trace(state: PlanningState, node_name: str) -> list[str]:
    return [*state.get("trace", []), node_name]


def _catalog_path_for(design_input: DesignInput) -> str:
    return design_input.preferences.catalog or design_input.preferences.catalog_id or "catalog.json"


def validate_input_node(state: PlanningState) -> PlanningState:
    raw_input = state.get("raw_input")
    if raw_input is None:
        raw_input = state.get("input")
    if raw_input is None:
        raise ValueError("Planning graph requires raw_input or input.")

    design_input = (
        raw_input if isinstance(raw_input, DesignInput) else DesignInput.model_validate(raw_input)
    )
    return {
        "input": design_input,
        "prompt": design_input.preferences.prompt,
        "catalog_path": _catalog_path_for(design_input),
        "trace": _trace(state, "validate_input"),
    }


def extract_intent_node(state: PlanningState) -> PlanningState:
    design_input = state["input"]
    intent = extract_intent(design_input, state.get("llm"))
    return {
        "intent": intent,
        "trace": _trace(state, "extract_intent"),
    }


def _categories_for_items(items: Iterable[str]) -> set[str]:
    categories: set[str] = set()
    for item in items:
        normalized = item.casefold()
        if "sink" in normalized:
            categories.add("sinks")
        elif normalized in {"tap", "faucet", "fixture"}:
            categories.add("fixtures")
        else:
            categories.add("appliances")
    return categories


def infer_retrieval_categories(intent: StructuredIntent) -> list[str]:
    categories: set[str] = {"cabinets"}
    categories.update(_categories_for_items(intent.required_items))

    for request in intent.color_requests:
        if request.target in {"cabinets", "base_cabinets", "wall_cabinets", "tall_cabinets"}:
            categories.add("cabinets")
        elif request.target == "appliances":
            categories.add("appliances")
        elif request.target == "sinks":
            categories.add("sinks")
        elif request.target == "fixtures":
            categories.add("fixtures")

    if intent.pantry_storage or intent.tall_cabinets:
        categories.add("cabinets")

    ordered = ["cabinets", "appliances", "sinks", "fixtures"]
    return [category for category in ordered if category in categories]


def _query_for_category(
    category: str,
    intent: StructuredIntent,
) -> CatalogRetrievalQuery:
    style = intent.style
    color = None
    if category == "cabinets" and intent.cabinet_color:
        color = intent.cabinet_color.resolved_hex or intent.cabinet_color.requested_hex
    return CatalogRetrievalQuery(category=category, style=style, color=color, limit=100)


async def retrieve_catalog_node(state: PlanningState) -> PlanningState:
    catalog = CatalogService.load(state.get("catalog_path", "catalog.json"))
    resolved_intent = resolve_intent_colors(catalog, state["intent"])
    categories = infer_retrieval_categories(resolved_intent)
    retriever = AsyncCatalogRetriever(catalog)
    results = await retriever.retrieve_many(
        _query_for_category(category, resolved_intent) for category in categories
    )
    payloads = {result.category: result.to_payload() for result in results}
    return {
        "catalog": catalog,
        "intent": resolved_intent,
        "retrieval_categories": categories,
        "retrieval_results": payloads,
        "trace": _trace(state, "retrieve_catalog"),
    }


def analyze_feasibility_node(state: PlanningState) -> PlanningState:
    result = analyze_feasibility(
        design_input=state["input"],
        intent=state["intent"],
        catalog=state["catalog"],
    )
    return {
        "feasibility": result.to_payload(),
        "trace": _trace(state, "analyze_feasibility"),
    }


def generate_variants_node(state: PlanningState) -> PlanningState:
    intent = state["intent"]
    feasibility = state.get("feasibility", {})
    family = feasibility.get("selected_family") or intent.layout_family
    templates = generate_topology_templates(feasibility)
    variants: list[dict[str, object]] = []
    for index, template in enumerate(templates, start=1):
        zone_plan = plan_zones_for_template(template, intent)
        placement_plan = generate_placement_plan(
            state["input"].environment,
            template,
            zone_plan,
            state["catalog"],
        )
        variants.append(
            {
                "id": f"variant-{template.family.lower()}-{index}",
                "family": template.family,
                "family_label": template.family_label,
                "status": "placed_template",
                "template_id": template.id,
                "topology": template.to_payload(),
                "zone_plan": zone_plan.to_payload(),
                "placement": placement_plan.to_payload(),
                "layout": placement_plan.layout_payload(),
                "notes": "Continuous snapped cabinet and appliance runs generated.",
            }
        )
    if not variants and family is not None:
        variants = [
            {
                "id": "variant-template-unavailable",
                "family": family,
                "status": "template_unavailable",
                "notes": "No procedural topology template could be derived from feasibility.",
            }
        ]
    return {
        "variants": variants,
        "trace": _trace(state, "generate_variants"),
    }


def validate_variants_node(state: PlanningState) -> PlanningState:
    validation_results = validate_variants(
        state["input"].environment,
        state.get("variants", []),
    )
    violations_by_variant = {
        result.variant_id: [
            violation.to_payload()
            for violation in result.violations
        ]
        for result in validation_results
    }
    variants = []
    for variant in state.get("variants", []):
        variant_with_violations = dict(variant)
        variant_with_violations["violations"] = violations_by_variant.get(
            str(variant.get("id") or ""),
            [],
        )
        variants.append(variant_with_violations)
    return {
        "variants": variants,
        "violations": flatten_validation_results(validation_results),
        "trace": _trace(state, "validate"),
    }


def repair_variants_node(state: PlanningState) -> PlanningState:
    repair_results = repair_variants(
        state["input"].environment,
        state["catalog"],
        state.get("variants", []),
    )
    return {
        "variants": [result.variant for result in repair_results],
        "repairs": flatten_repair_actions(repair_results),
        "violations": flatten_repair_violations(repair_results),
        "trace": _trace(state, "repair"),
    }


def score_variants_node(state: PlanningState) -> PlanningState:
    variants = state.get("variants", [])
    return {
        "scores": [
            {
                "variant_id": str(variant.get("id", "")),
                "score": 0.0,
                "status": "unscored_skeleton",
            }
            for variant in variants
        ],
        "trace": _trace(state, "score"),
    }


def assemble_output_node(state: PlanningState) -> PlanningState:
    feasibility = state.get("feasibility", {})
    return {
        "output": {
            "status": "skeleton",
            "prompt": state.get("prompt", ""),
            "layout_family": feasibility.get("selected_family") or state["intent"].layout_family,
            "feasibility_status": feasibility.get("status"),
            "retrieval_categories": state.get("retrieval_categories", []),
            "variant_count": len(state.get("variants", [])),
            "ready_for_render": False,
        },
        "trace": _trace(state, "assemble_output"),
    }


def build_planning_graph() -> Any:
    graph = StateGraph(PlanningState)
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("extract_intent", extract_intent_node)
    graph.add_node("retrieve_catalog", retrieve_catalog_node)
    graph.add_node("analyze_feasibility", analyze_feasibility_node)
    graph.add_node("generate_variants", generate_variants_node)
    graph.add_node("validate", validate_variants_node)
    graph.add_node("repair", repair_variants_node)
    graph.add_node("score", score_variants_node)
    graph.add_node("assemble_output", assemble_output_node)

    graph.add_edge(START, "validate_input")
    for source, target in zip(GRAPH_NODE_ORDER[:-1], GRAPH_NODE_ORDER[1:], strict=True):
        graph.add_edge(source, target)
    graph.add_edge("assemble_output", END)
    return graph.compile()


async def run_planning_graph_async(initial_state: PlanningState) -> PlanningState:
    graph = build_planning_graph()
    result = await graph.ainvoke(initial_state)
    return result


def run_planning_graph(initial_state: PlanningState) -> PlanningState:
    return asyncio.run(run_planning_graph_async(initial_state))
