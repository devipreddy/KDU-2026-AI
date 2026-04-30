"""CLI entrypoint for the five-phase lab."""

from __future__ import annotations

import argparse
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from .logging_utils import configure_logging
from .schemas import to_plain_data
from .settings import AppSettings
from .workflows.memory_demo import run_memory_demo
from .workflows.orchestration import run_coordination_query
from .workflows.phase1 import run_phase1
from .workflows.planner_executor import run_planner_executor


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain_data(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AgentKit orchestration lab.")
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Project root used for local storage.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("phase1", help="Run the loop detection demo.")

    coord = subparsers.add_parser("phase2", help="Run the coordinator isolation demo.")
    coord.add_argument(
        "--input",
        default="What is John's salary and how much PTO does he have?",
        help="User input for the coordination workflow.",
    )

    phase3 = subparsers.add_parser("phase3", help="Run the structured context handoff demo.")
    phase3.add_argument(
        "--input",
        default="Update my banking details. Routing number is 123456789.",
        help="User input for the structured context handoff workflow.",
    )

    subparsers.add_parser("phase4", help="Run the memory compaction demo.")

    phase5 = subparsers.add_parser("phase5", help="Run the planner-executor demo.")
    phase5.add_argument(
        "--input",
        default="What is John's salary and how much PTO does he have?",
        help="User request for the planner-executor workflow.",
    )
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    settings = AppSettings.from_env(args.project_root)

    if args.command == "phase1":
        result = run_phase1(settings)
    elif args.command == "phase2":
        result = run_coordination_query(settings, args.input, session_id="phase2-demo")
    elif args.command == "phase3":
        result = run_coordination_query(settings, args.input, session_id="phase3-demo")
    elif args.command == "phase4":
        result = run_memory_demo(settings)
    elif args.command == "phase5":
        result = run_planner_executor(settings, args.input)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(_serialize(result), indent=2))


if __name__ == "__main__":
    main()
