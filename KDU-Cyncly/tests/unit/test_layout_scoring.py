from __future__ import annotations

from pathlib import Path

from auto_design.catalog.service import CatalogService
from auto_design.scoring import CONFIDENCE_WEIGHTS, rank_variants, score_variant
from auto_design.schemas import ColorRequest, StructuredIntent


ROOT = Path(__file__).resolve().parents[2]


def load_catalog() -> CatalogService:
    return CatalogService.load(ROOT / "catalog.json")


def base_variant(
    variant_id: str,
    *,
    product_id: str = "SKU-C11",
    violations: list[dict[str, object]] | None = None,
    repair_history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": variant_id,
        "family": "L",
        "status": "placed_template",
        "placement": {
            "family": "L",
            "is_continuous": True,
            "base_coverage_valid": True,
            "runs": [
                {
                    "wall": "north",
                    "run_role": "primary",
                    "start_mm": 0.0,
                    "end_mm": 900.0,
                    "items": [
                        {
                            "key": f"{variant_id}_base",
                            "component": "base_cabinet",
                            "product_id": product_id,
                            "product_type": "base_cabinet",
                            "wall": "north",
                            "run_role": "primary",
                            "zone_type": "storage",
                            "layer": "base_run",
                            "start_mm": 0.0,
                            "end_mm": 900.0,
                            "position_mm": {"x": 450.0, "y": 0.0, "z": 450.0},
                            "dimensions_mm": {
                                "width": 900.0,
                                "depth": 600.0,
                                "height": 900.0,
                            },
                            "rotation_z_deg": 0.0,
                            "is_base_cabinet": True,
                        }
                    ],
                    "continuity_gaps_mm": [],
                    "is_continuous": True,
                    "starts_with_base": True,
                    "ends_with_base": True,
                }
            ],
            "overhead_items": [],
            "base_coverages": [],
        },
        "layout": {
            f"{variant_id}_base": {
                "product_id": product_id,
                "position_mm": {"x": 450.0, "y": 0.0, "z": 450.0},
                "dimensions_mm": {"width": 900.0, "depth": 600.0, "height": 900.0},
                "rotation_z_deg": 0.0,
                "anchor_wall": "north",
                "zone_type": "storage",
            }
        },
        "violations": violations or [],
        "repair_history": repair_history or [],
    }


def test_clean_layout_gets_full_composite_confidence() -> None:
    score = score_variant(
        base_variant("clean"),
        intent=StructuredIntent(layout_family="L"),
        catalog=load_catalog(),
        feasibility={"selected_family": "L", "status": "feasible"},
    )

    assert CONFIDENCE_WEIGHTS == {
        "geometry": 0.4,
        "workflow": 0.3,
        "retrieval_color": 0.2,
        "topology": 0.1,
    }
    assert score.score == 1.0
    assert score.confidence == {
        "geometry": 1.0,
        "workflow": 1.0,
        "retrieval_color": 1.0,
        "topology": 1.0,
    }
    assert score.status == "ranked_clean"


def test_ranking_preserves_flagged_but_renderable_variants() -> None:
    clean = base_variant("clean")
    flagged = base_variant(
        "flagged",
        violations=[
            {
                "rule_id": "WORKFLOW-01",
                "severity": "soft",
                "text": "Dishwasher is too far from sink.",
            }
        ],
    )

    ranked, scores = rank_variants(
        [flagged, clean],
        intent=StructuredIntent(layout_family="L"),
        catalog=load_catalog(),
        feasibility={"selected_family": "L", "status": "feasible"},
    )

    assert [variant["id"] for variant in ranked] == ["clean", "flagged"]
    assert [score["variant_id"] for score in scores] == ["clean", "flagged"]
    assert scores[1]["renderable"] is True
    assert scores[1]["status"] == "ranked_with_flags"
    assert ranked[1]["violations"]


def test_color_retrieval_confidence_rewards_matching_catalog_skus() -> None:
    color = ColorRequest(
        raw_text="navy blue",
        target="base_cabinets",
        requested_hex="#1F3A5F",
        resolved_hex="#1F3A5F",
        matched_skus=["SKU-C11"],
    )
    intent = StructuredIntent(
        layout_family="L",
        color_requests=[color],
        cabinet_color=color,
    )
    catalog = load_catalog()

    navy_score = score_variant(
        base_variant("navy", product_id="SKU-C11"),
        intent=intent,
        catalog=catalog,
        feasibility={"selected_family": "L", "status": "feasible"},
    )
    oak_score = score_variant(
        base_variant("oak", product_id="SKU-C07"),
        intent=intent,
        catalog=catalog,
        feasibility={"selected_family": "L", "status": "feasible"},
    )

    assert navy_score.confidence["retrieval_color"] == 1.0
    assert oak_score.confidence["retrieval_color"] == 0.0
    assert navy_score.score > oak_score.score


def test_hard_geometry_violation_lowers_geometry_weight_only() -> None:
    scored = score_variant(
        base_variant(
            "broken",
            violations=[
                {
                    "rule_id": "LAYOUT-03",
                    "severity": "hard",
                    "text": "Continuity gap is too large.",
                }
            ],
        ),
        intent=StructuredIntent(layout_family="L"),
        catalog=load_catalog(),
        feasibility={"selected_family": "L", "status": "feasible"},
    )

    assert scored.confidence["geometry"] == 0.7
    assert scored.confidence["workflow"] == 1.0
    assert scored.score == 0.88
    assert scored.renderable is True
