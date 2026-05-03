from __future__ import annotations

import json
from typing import Literal

from autonomous_research_department.runtime import bootstrap_environment

bootstrap_environment()

from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.flow.persistence import persist

from autonomous_research_department.crew import ResearchDepartmentCrew
from autonomous_research_department.models import FactCheckDecision, FlowStateModel
from autonomous_research_department.settings import get_settings


def decide_route(
    decision: FactCheckDecision,
    retry_count: int,
    max_retries: int,
) -> Literal["write", "retry", "halt"]:
    if decision.route == "write" and decision.verdict == "approved":
        return "write"
    if decision.route == "retry" and retry_count < max_retries:
        return "retry"
    return "halt"


@persist()
class ResearchDepartmentFlow(Flow[FlowStateModel]):
    """Stateful Flow that explicitly routes based on fact-check output."""

    def _prepare_task(self, task, inputs: dict[str, str]) -> None:
        task.interpolate_inputs_and_add_conversation_history(inputs)

    @start()
    def initialize(self) -> str:
        if not self.state.max_retries:
            self.state.max_retries = get_settings().flow_max_retries
        self.remember(
            f"Started research flow for topic '{self.state.topic}'",
            scope="/flow/runs",
        )
        return self.state.topic

    @listen(initialize)
    def research(self) -> str:
        department = ResearchDepartmentCrew()
        task = department.research_task()
        self._prepare_task(task, {"topic": self.state.topic})
        output = task.execute_sync(agent=department.researcher())
        self.state.research_output = output.raw or ""
        self.remember(
            f"Initial research completed for '{self.state.topic}'",
            scope="/flow/research",
        )
        return self.state.research_output

    @listen("retry")
    def retry_research(self) -> str:
        department = ResearchDepartmentCrew()
        self.state.retry_count += 1
        guidance = ""
        if self.state.fact_check is not None:
            guidance = (
                "Revise the research using this fact-check feedback:\n"
                f"{self.state.fact_check.model_dump_json(indent=2)}"
            )

        task = department.research_task()
        self._prepare_task(task, {"topic": self.state.topic})
        output = task.execute_sync(
            agent=department.researcher(),
            context=guidance,
        )
        self.state.research_output = output.raw or ""
        self.remember(
            f"Retry {self.state.retry_count} completed for '{self.state.topic}'",
            scope="/flow/retries",
        )
        return self.state.research_output

    @listen(or_(research, retry_research))
    def fact_check(self, _: str | None = None) -> dict[str, str]:
        department = ResearchDepartmentCrew()
        output = department.fact_check_task().execute_sync(
            agent=department.fact_checker(),
            context=self.state.research_output,
        )

        if output.pydantic is None:
            raise RuntimeError(
                "Fact-check task did not return a structured FactCheckDecision."
            )

        self.state.fact_check = FactCheckDecision.model_validate(
            output.pydantic.model_dump()  # type: ignore[union-attr]
        )
        self.remember(
            f"Fact-check route for '{self.state.topic}': {self.state.fact_check.route}",
            scope="/flow/fact-check",
        )
        return {"route": self.state.fact_check.route}

    @router(fact_check)
    def route_after_fact_check(self) -> Literal["write", "retry", "halt"]:
        if self.state.fact_check is None:
            self.state.halt_reason = "Fact-check produced no structured decision."
            route: Literal["write", "retry", "halt"] = "halt"
        else:
            route = decide_route(
                decision=self.state.fact_check,
                retry_count=self.state.retry_count,
                max_retries=self.state.max_retries,
            )

        self.state.route_history.append(route)
        if route == "halt" and self.state.halt_reason is None and self.state.fact_check:
            self.state.halt_reason = (
                "Retry budget exhausted or fact-check marked the brief unsafe to publish."
            )
        return route

    @listen("write")
    def write_brief(self) -> str:
        department = ResearchDepartmentCrew()
        context = "\n\n".join(
            [
                "Research dossier:",
                self.state.research_output,
                "Fact-check decision:",
                json.dumps(
                    self.state.fact_check.model_dump() if self.state.fact_check else {},
                    indent=2,
                ),
            ]
        )
        output = department.writing_task().execute_sync(
            agent=department.writer(),
            context=context,
        )
        self.state.final_output = output.raw or ""
        self.remember(
            f"Final brief written for '{self.state.topic}'",
            scope="/flow/final",
        )
        return self.state.final_output

    @listen("halt")
    def halt_workflow(self) -> str:
        decision = (
            self.state.fact_check.model_dump_json(indent=2)
            if self.state.fact_check
            else "{}"
        )
        self.state.final_output = "\n".join(
            [
                "# Workflow Halted",
                "",
                f"Topic: {self.state.topic}",
                f"Reason: {self.state.halt_reason or 'Workflow was halted by router logic.'}",
                "",
                "## Latest Fact-Check Decision",
                "```json",
                decision,
                "```",
            ]
        )
        return self.state.final_output
