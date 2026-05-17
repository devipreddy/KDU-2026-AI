from __future__ import annotations

from auto_design.catalog.service import CatalogService
from auto_design.planner import analyze_feasibility, generate_topology_templates
from auto_design.schemas import DesignInput, StructuredIntent


def room_payload(
    *,
    width_mm: int = 4200,
    depth_mm: int = 3200,
    cabinet_walls: set[str] | None = None,
) -> dict[str, object]:
    cabinet_walls = cabinet_walls or {"north", "south", "east", "west"}
    wall_specs = [
        ("south_wall", "south", width_mm),
        ("north_wall", "north", width_mm),
        ("east_wall", "east", depth_mm),
        ("west_wall", "west", depth_mm),
    ]
    points = {
        "south": [
            {"x": 0, "y": 0, "z": 0},
            {"x": width_mm, "y": 0, "z": 0},
            {"x": width_mm, "y": 0, "z": 2700},
            {"x": 0, "y": 0, "z": 2700},
        ],
        "north": [
            {"x": 0, "y": depth_mm, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 2700},
            {"x": 0, "y": depth_mm, "z": 2700},
        ],
        "east": [
            {"x": width_mm, "y": 0, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 0},
            {"x": width_mm, "y": depth_mm, "z": 2700},
            {"x": width_mm, "y": 0, "z": 2700},
        ],
        "west": [
            {"x": 0, "y": 0, "z": 0},
            {"x": 0, "y": depth_mm, "z": 0},
            {"x": 0, "y": depth_mm, "z": 2700},
            {"x": 0, "y": 0, "z": 2700},
        ],
    }
    return {
        "environment": {
            "floor": {
                "points": [
                    {"x": 0, "y": 0, "z": 0},
                    {"x": width_mm, "y": 0, "z": 0},
                    {"x": width_mm, "y": depth_mm, "z": 0},
                    {"x": 0, "y": depth_mm, "z": 0},
                ]
            },
            "wall": [
                {
                    "name": name,
                    "anchor": anchor,
                    "thickness_mm": 100,
                    "has_cabinets": anchor in cabinet_walls,
                    "dimensions": {"length_mm": length, "height": 2700},
                    "points": points[anchor],
                }
                for name, anchor, length in wall_specs
            ],
            "openings": [],
        },
        "preferences": {
            "budget_tier": "mid",
            "must_have": ["dishwasher", "hood"],
            "avoid": [],
            "prompt": "",
            "catalog": "./catalog.json",
        },
    }


def feasibility_payload_for(family: str) -> dict[str, object]:
    design_input = DesignInput.model_validate(room_payload())
    intent = StructuredIntent(
        layout_family=family,
        required_items=design_input.preferences.must_have,
    )
    catalog = CatalogService.load("catalog.json")
    return analyze_feasibility(design_input, intent, catalog).to_payload()


def test_single_wall_grammar_generates_only_i_family_templates() -> None:
    templates = generate_topology_templates(feasibility_payload_for("I"))

    assert len(templates) == 3
    assert {template.family for template in templates} == {"I"}
    assert all(len(template.walls) == 1 for template in templates)
    assert all(len(template.runs) == 1 for template in templates)
    assert all(template.runs[0].role == "primary" for template in templates)


def test_l_shape_grammar_generates_only_l_family_templates() -> None:
    templates = generate_topology_templates(feasibility_payload_for("L"))

    assert len(templates) == 3
    assert {template.family for template in templates} == {"L"}
    assert all(len(template.walls) == 2 for template in templates)
    assert all(
        [run.role for run in template.runs] == ["primary", "return"]
        for template in templates
    )
    assert all(
        {"cooling", "cleaning", "cooking"}.issubset(
            {step.zone_type for step in template.steps}
        )
        for template in templates
    )


def test_u_shape_grammar_generates_only_u_family_templates() -> None:
    templates = generate_topology_templates(feasibility_payload_for("U"))

    assert len(templates) == 3
    assert {template.family for template in templates} == {"U"}
    assert all(len(template.walls) == 3 for template in templates)
    assert all(
        [run.role for run in template.runs] == ["left_leg", "bridge", "right_leg"]
        for template in templates
    )


def test_template_payload_contains_serializable_runs_and_steps() -> None:
    template = generate_topology_templates(feasibility_payload_for("L"))[0]
    payload = template.to_payload()

    assert payload["family"] == "L"
    assert payload["family_label"] == "L-shaped"
    assert len(payload["runs"]) == 2
    assert len(payload["steps"]) == 5
    assert payload["steps"][0]["order"] == 1
