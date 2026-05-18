from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from auto_design.agent import run_planning_graph
from auto_design.schemas import LayoutResponse


def _default_render_script() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "render.py"
    if candidate.exists():
        return candidate
    return Path.cwd() / "render.py"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"input file is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("input file must contain a JSON object")
    return payload


def _apply_overrides(
    payload: dict[str, Any],
    *,
    prompt: str | None,
    catalog: str | None,
) -> dict[str, Any]:
    updated = dict(payload)
    preferences = updated.get("preferences")
    if not isinstance(preferences, dict):
        preferences = {}
    else:
        preferences = dict(preferences)

    if prompt is not None:
        preferences["prompt"] = prompt
    if catalog is not None:
        preferences["catalog"] = catalog
    updated["preferences"] = preferences
    return updated


def _write_output(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generate_output(
    input_path: Path,
    *,
    prompt: str | None,
    catalog: str | None,
) -> dict[str, object]:
    raw_input = _load_json(input_path)
    raw_input = _apply_overrides(
        raw_input,
        prompt=prompt,
        catalog=catalog,
    )
    result = run_planning_graph({"raw_input": raw_input})
    return LayoutResponse.model_validate(result["output"]).model_dump(
        mode="json",
        exclude_none=True,
    )


def generate_command(args: argparse.Namespace) -> int:
    output = _generate_output(
        args.input,
        prompt=args.prompt,
        catalog=args.catalog,
    )
    _write_output(args.output, output)
    print(
        f"Generated {len(output['layouts'])} layout(s) -> {args.output}",
        file=sys.stderr,
    )
    return 0


def generate_render_command(args: argparse.Namespace) -> int:
    output = _generate_output(
        args.input,
        prompt=args.prompt,
        catalog=args.catalog,
    )
    _write_output(args.output, output)
    print(
        f"Generated {len(output['layouts'])} layout(s) -> {args.output}",
        file=sys.stderr,
    )

    render_script = args.render_script or _default_render_script()
    if not render_script.exists():
        raise ValueError(f"render.py not found: {render_script}")

    command = [
        sys.executable,
        str(render_script),
        str(args.output),
        "--strict",
        "--out-dir",
        str(args.out_dir),
    ]
    if args.catalog is not None:
        command.extend(["--catalog", args.catalog])
    if args.show:
        command.append("--show")
    if args.two_d_only:
        command.append("--2d-only")
    if args.three_d_only:
        command.append("--3d-only")

    print(f"Rendering with: {' '.join(command)}", file=sys.stderr)
    return subprocess.run(command, check=False).returncode


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input.json room/environment request.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional natural-language prompt override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where renderer-compatible output.json should be written.",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Optional catalog path override stored in input preferences.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auto_design",
        description="Auto-design kitchen layout generation tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Run the LangGraph planner and write renderer-compatible output JSON.",
    )
    _add_generation_arguments(generate)
    generate.set_defaults(func=generate_command)

    generate_render = subparsers.add_parser(
        "generate-render",
        help="Generate output JSON, then render it through the existing render.py.",
    )
    _add_generation_arguments(generate_render)
    generate_render.add_argument(
        "--out-dir",
        type=Path,
        default=Path("renders"),
        help="Directory passed to render.py for PNG output.",
    )
    generate_render.add_argument(
        "--render-script",
        type=Path,
        default=None,
        help="Optional path to the existing render.py script.",
    )
    generate_render.add_argument(
        "--show",
        action="store_true",
        help="Pass --show through to render.py for interactive 3D display.",
    )
    render_mode = generate_render.add_mutually_exclusive_group()
    render_mode.add_argument(
        "--2d-only",
        dest="two_d_only",
        action="store_true",
        help="Pass --2d-only through to render.py.",
    )
    render_mode.add_argument(
        "--3d-only",
        dest="three_d_only",
        action="store_true",
        help="Pass --3d-only through to render.py.",
    )
    generate_render.set_defaults(func=generate_render_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
