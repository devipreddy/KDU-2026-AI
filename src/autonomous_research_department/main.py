from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from autonomous_research_department.runtime import bootstrap_environment

bootstrap_environment()
load_dotenv()

from crewai.crews.crew_output import CrewOutput

from autonomous_research_department.crew import ResearchDepartmentCrew
from autonomous_research_department.flow import ResearchDepartmentFlow
from autonomous_research_department.models import RunArtifact, TaskSnapshot
from autonomous_research_department.settings import AppSettings, get_settings


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _serialize_task_output(output: CrewOutput) -> list[TaskSnapshot]:
    snapshots: list[TaskSnapshot] = []
    for task_output in output.tasks_output:
        snapshots.append(
            TaskSnapshot(
                name=task_output.name,
                description=task_output.description,
                expected_output=task_output.expected_output,
                agent=task_output.agent,
                raw=task_output.raw,
                pydantic=(
                    task_output.pydantic.model_dump()
                    if task_output.pydantic is not None
                    else None
                ),
            )
        )
    return snapshots


def _artifact_path(settings: AppSettings, mode: str) -> Path:
    return settings.outputs_dir / f"{_timestamp()}-{mode}.json"


def _run_crew(mode: str, topic: str, settings: AppSettings) -> RunArtifact:
    department = ResearchDepartmentCrew(settings=settings)
    crew = (
        department.build_sequential_crew()
        if mode == "sequential"
        else department.build_hierarchical_crew()
    )
    output = crew.kickoff(inputs={"topic": topic})
    token_usage = None
    if output.token_usage is not None:
        if hasattr(output.token_usage, "model_dump"):
            token_usage = output.token_usage.model_dump()
        elif isinstance(output.token_usage, dict):
            token_usage = output.token_usage

    return RunArtifact(
        mode=mode,
        topic=topic,
        final_output=output.raw,
        tasks=_serialize_task_output(output),
        token_usage=token_usage,
    )


def run_sequential(topic: str, settings: AppSettings) -> RunArtifact:
    return _run_crew("sequential", topic, settings)


def run_hierarchical(topic: str, settings: AppSettings) -> RunArtifact:
    return _run_crew("hierarchical", topic, settings)


def run_flow(topic: str, settings: AppSettings) -> RunArtifact:
    flow = ResearchDepartmentFlow()
    result = flow.kickoff(inputs={"topic": topic, "max_retries": settings.flow_max_retries})
    return RunArtifact(
        mode="flow",
        topic=topic,
        final_output=str(result),
        tasks=[],
        token_usage=None,
    )


def inspect_memory(settings: AppSettings) -> str:
    from crewai import Memory

    memory = Memory(
        llm=settings.primary_llm_config().model,
        storage=str(settings.memory_dir),
    )
    return memory.tree()


def _print_and_store(artifact: RunArtifact, settings: AppSettings) -> None:
    path = _artifact_path(settings, artifact.mode)
    artifact.write_json(path)
    print(path)
    print()
    print(artifact.final_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CrewAI autonomous research department lab runner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("sequential", "hierarchical", "flow", "compare"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "--topic",
            required=True,
            help="Research topic to run through the department.",
        )

    subparsers.add_parser("inspect-memory")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "inspect-memory":
        print(inspect_memory(settings))
        return

    settings.validate_required_keys()

    if args.command == "sequential":
        artifact = run_sequential(args.topic, settings)
        _print_and_store(artifact, settings)
        return

    if args.command == "hierarchical":
        artifact = run_hierarchical(args.topic, settings)
        _print_and_store(artifact, settings)
        return

    if args.command == "flow":
        artifact = run_flow(args.topic, settings)
        _print_and_store(artifact, settings)
        return

    if args.command == "compare":
        sequential = run_sequential(args.topic, settings)
        hierarchical = run_hierarchical(args.topic, settings)
        sequential.write_json(_artifact_path(settings, "sequential"))
        hierarchical.write_json(_artifact_path(settings, "hierarchical"))
        print("Sequential final output:")
        print(sequential.final_output)
        print()
        print("Hierarchical final output:")
        print(hierarchical.final_output)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
