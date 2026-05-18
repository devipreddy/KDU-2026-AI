from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from auto_design.agent import run_planning_graph
from auto_design.schemas import LayoutResponse


ROOT = Path(__file__).resolve().parents[2]
NAVY_HEX = "#1F3A5F"


def load_input(name: str) -> dict[str, object]:
    payload = json.loads((ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def catalog_by_sku() -> dict[str, dict[str, object]]:
    products = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    return {str(product["id"]): product for product in products}


def enable_cabinet_walls(payload: dict[str, object], anchors: set[str]) -> None:
    environment = payload["environment"]
    assert isinstance(environment, dict)
    walls = environment["wall"]
    assert isinstance(walls, list)
    for wall in walls:
        assert isinstance(wall, dict)
        wall["has_cabinets"] = wall.get("anchor") in anchors


def product_ids(output: dict[str, object]) -> Iterable[str]:
    for layout in output["layouts"]:
        for item in layout["layout"].values():
            sku = item.get("product_id")
            if sku:
                yield str(sku)


def render_output(output: dict[str, object], tmp_path: Path, topology: str) -> None:
    output_path = tmp_path / f"{topology.lower()}-output.json"
    output_path.write_text(json.dumps(output), encoding="utf-8")
    out_dir = tmp_path / f"{topology.lower()}-renders"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "render.py"),
            str(output_path),
            "--out-dir",
            str(out_dir),
            "--catalog",
            str(ROOT / "catalog.json"),
            "--strict",
            "--2d-only",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for layout in output["layouts"]:
        assert (out_dir / f"{layout['id']}_top.png").is_file()


@pytest.mark.parametrize(
    ("topology", "input_name", "prompt", "expected_family", "cabinet_walls"),
    [
        (
            "I",
            "input1.json",
            "navy blue single-wall kitchen with dishwasher hood no uppers",
            "I-shaped",
            None,
        ),
        (
            "L",
            "input3.json",
            "navy blue L-shape kitchen with dishwasher hood no uppers",
            "L-shaped",
            None,
        ),
        (
            "U",
            "input2.json",
            "navy blue U-shape kitchen with dishwasher hood no uppers",
            "U-shaped",
            {"north", "east", "south"},
        ),
    ],
)
def test_e2e_topology_fixtures_generate_validate_and_render(
    topology: str,
    input_name: str,
    prompt: str,
    expected_family: str,
    cabinet_walls: set[str] | None,
    tmp_path: Path,
) -> None:
    payload = load_input(input_name)
    if cabinet_walls is not None:
        enable_cabinet_walls(payload, cabinet_walls)
    assert isinstance(payload["preferences"], dict)
    payload["preferences"]["prompt"] = prompt

    result = run_planning_graph({"raw_input": payload})
    output = result["output"]
    LayoutResponse.model_validate(output)

    assert 3 <= len(output["layouts"]) <= 5
    assert {layout["family"] for layout in output["layouts"]} == {expected_family}

    catalog = catalog_by_sku()
    assert set(product_ids(output)).issubset(catalog)
    for layout in output["layouts"]:
        layout_skus = [
            str(item["product_id"])
            for item in layout["layout"].values()
            if item.get("product_id")
        ]
        assert "SKU-C11" in layout_skus
        for sku in layout_skus:
            product = catalog[sku]
            product_type = str(product["type"])
            assert not product_type.startswith(("wall_", "tall_"))
            if product["category"] == "cabinet" and product_type.startswith("base"):
                assert product["color"].upper() == NAVY_HEX
        assert any(
            rationale["rule_id"] == "COLOR-01"
            and "SKU-C11" in rationale["text"]
            for rationale in layout["rationale"]
        )

    render_output(output, tmp_path, topology)
