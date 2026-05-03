from __future__ import annotations

import random
from typing import ClassVar

from pydantic import BaseModel, Field

from autonomous_research_department.runtime import bootstrap_environment

bootstrap_environment()

from crewai.tools import BaseTool


class IntermittentProbeInput(BaseModel):
    query: str = Field(..., description="Research query to probe.")


class IntermittentResearchProbeTool(BaseTool):
    name: str = "intermittent_research_probe"
    description: str = (
        "A flaky research probe that sometimes times out. "
        "Use it to simulate unreliable upstream tooling."
    )
    args_schema: type[BaseModel] = IntermittentProbeInput

    failure_rate: float = 0.5
    seed: int | None = None
    _shared_rng: ClassVar[random.Random | None] = None

    def _rng(self) -> random.Random:
        if self.seed is None:
            return random.Random()
        if self.__class__._shared_rng is None:
            self.__class__._shared_rng = random.Random(self.seed)
        return self.__class__._shared_rng

    def _run(self, query: str) -> str:
        if self._rng().random() < self.failure_rate:
            raise TimeoutError(
                "Simulated probe timeout after 30 seconds while fetching supplemental evidence."
            )

        return (
            "Probe status: success\n"
            f"Query: {query}\n"
            "Supplemental signal: public discourse appears mixed, so claims should be cross-checked with stronger sources."
        )

