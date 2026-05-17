from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


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
