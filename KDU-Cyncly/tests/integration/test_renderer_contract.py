from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from auto_design.agent import run_planning_graph


ROOT = Path(__file__).resolve().parents[2]


def test_render_cli_accepts_output_envelope_and_known_catalog_skus(tmp_path: Path) -> None:
    sample_output = json.loads((ROOT / "output.json").read_text(encoding="utf-8"))
    envelope = {
        "request_id": "renderer-contract-test",
        "duration_ms": 1,
        "layouts": [sample_output["layouts"][0]],
    }
    response_path = tmp_path / "output.json"
    response_path.write_text(json.dumps(envelope), encoding="utf-8")

    out_dir = tmp_path / "renders"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "render.py"),
            str(response_path),
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
    assert (out_dir / "variant-1_top.png").is_file()


def test_render_cli_accepts_generated_graph_output_envelope(tmp_path: Path) -> None:
    payload = json.loads((ROOT / "input3.json").read_text(encoding="utf-8"))
    graph_output = run_planning_graph({"raw_input": payload})["output"]
    envelope = {
        "request_id": graph_output["request_id"],
        "duration_ms": graph_output["duration_ms"],
        "layouts": [graph_output["layouts"][0]],
    }
    response_path = tmp_path / "generated-output.json"
    response_path.write_text(json.dumps(envelope), encoding="utf-8")

    out_dir = tmp_path / "renders"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "render.py"),
            str(response_path),
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

    variant_id = str(envelope["layouts"][0]["id"])
    assert result.returncode == 0, result.stderr
    assert (out_dir / f"{variant_id}_top.png").is_file()
