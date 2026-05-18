from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from auto_design.schemas import LayoutResponse


ROOT = Path(__file__).resolve().parents[2]


def _catalog_by_sku() -> dict[str, dict[str, object]]:
    products = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    return {str(product["id"]): product for product in products}


def _run_generate_cli(
    tmp_path: Path,
    prompt: str,
    *,
    input_name: str = "input3.json",
) -> subprocess.CompletedProcess[str]:
    output_path = tmp_path / "output.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_design",
            "generate",
            "--input",
            str(ROOT / input_name),
            "--prompt",
            prompt,
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_generate_render_cli(
    tmp_path: Path,
    prompt: str,
) -> subprocess.CompletedProcess[str]:
    output_path = tmp_path / "output.json"
    out_dir = tmp_path / "renders"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_design",
            "generate-render",
            "--input",
            str(ROOT / "input3.json"),
            "--prompt",
            prompt,
            "--output",
            str(output_path),
            "--catalog",
            str(ROOT / "catalog.json"),
            "--out-dir",
            str(out_dir),
            "--render-script",
            str(ROOT / "render.py"),
            "--2d-only",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_generate_cli_writes_renderer_compatible_output(tmp_path: Path) -> None:
    output_path = tmp_path / "output.json"
    result = _run_generate_cli(
        tmp_path,
        "navy blue L-shape with no uppers",
    )

    assert result.returncode == 0, result.stderr
    assert "Generated 5 layout(s)" in result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    LayoutResponse.model_validate(payload)
    assert {layout["family"] for layout in payload["layouts"]} == {"L-shaped"}
    assert payload["layouts"][0]["layout"]["north_wall"]["is_wall"] is True
    assert payload["layouts"][0]["layout"]["south_door"]["is_door"] is True

    catalog = _catalog_by_sku()
    for layout in payload["layouts"]:
        for item in layout["layout"].values():
            sku = item.get("product_id")
            if not sku:
                continue
            product_type = str(catalog[str(sku)]["type"])
            assert not product_type.startswith(("wall_", "tall_"))


def test_generate_cli_output_is_deterministic(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _run_generate_cli(
        first_dir,
        "single-wall kitchen with dishwasher and hood",
        input_name="input1.json",
    )
    second = _run_generate_cli(
        second_dir,
        "single-wall kitchen with dishwasher and hood",
        input_name="input1.json",
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_payload = json.loads((first_dir / "output.json").read_text(encoding="utf-8"))
    second_payload = json.loads((second_dir / "output.json").read_text(encoding="utf-8"))
    assert first_payload == second_payload
    assert {layout["family"] for layout in first_payload["layouts"]} == {"I-shaped"}


def test_generate_render_cli_calls_existing_renderer_strictly(tmp_path: Path) -> None:
    result = _run_generate_render_cli(
        tmp_path,
        "navy blue L-shape with no uppers",
    )

    assert result.returncode == 0, result.stderr
    assert "Generated 5 layout(s)" in result.stderr
    assert "render.py" in result.stderr
    assert "--strict" in result.stderr

    output_path = tmp_path / "output.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    LayoutResponse.model_validate(payload)

    first_variant_id = str(payload["layouts"][0]["id"])
    assert (tmp_path / "renders" / f"{first_variant_id}_top.png").is_file()
