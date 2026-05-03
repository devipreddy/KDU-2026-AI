from __future__ import annotations

from pathlib import Path

from autonomous_research_department.runtime import bootstrap_environment

bootstrap_environment()

from crewai import Agent, Crew, LLM, Memory, Process, Task
from crewai.project import CrewBase, agent, llm, output_pydantic, task, tool
from crewai_tools import SerperDevTool

from autonomous_research_department.models import FactCheckDecision
from autonomous_research_department.settings import AppSettings, get_settings
from autonomous_research_department.tools import IntermittentResearchProbeTool


@CrewBase
class ResearchDepartmentCrew:
    """YAML-configured CrewAI implementation for the lab."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    fact_check_decision = output_pydantic(FactCheckDecision)

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._shared_memory: Memory | None = None

    def shared_memory(self) -> Memory:
        if self._shared_memory is None:
            self._shared_memory = Memory(
                llm=self.settings.primary_llm_config().model,
                storage=str(self.settings.memory_dir),
            )
        return self._shared_memory

    @llm
    def primary_llm(self) -> LLM:
        config = self.settings.primary_llm_config()
        return LLM(
            model=config.model,
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            temperature=config.temperature,
        )

    @llm
    def manager_llm(self) -> LLM:
        config = self.settings.manager_llm_config()
        return LLM(
            model=config.model,
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            temperature=config.temperature,
        )

    @tool
    def serper_search(self) -> SerperDevTool:
        return SerperDevTool(n_results=5, search_type="search")

    @tool
    def intermittent_probe(self) -> IntermittentResearchProbeTool:
        return IntermittentResearchProbeTool(
            failure_rate=self.settings.intermittent_tool_failure_rate,
            seed=self.settings.intermittent_tool_seed,
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            memory=self.shared_memory(),
            max_iter=self.settings.agent_max_iter,
        )

    @agent
    def fact_checker(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_checker"],
            memory=self.shared_memory(),
            max_iter=self.settings.agent_max_iter,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            memory=self.shared_memory(),
            max_iter=self.settings.agent_max_iter,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def fact_check_task(self) -> Task:
        return Task(config=self.tasks_config["fact_check_task"])

    @task
    def writing_task(self) -> Task:
        return Task(config=self.tasks_config["writing_task"])

    def build_tasks(self) -> list[Task]:
        return [self.research_task(), self.fact_check_task(), self.writing_task()]

    def build_agents(self) -> list[Agent]:
        return [self.researcher(), self.fact_checker(), self.writer()]

    def build_sequential_crew(self) -> Crew:
        return Crew(
            agents=self.build_agents(),
            tasks=self.build_tasks(),
            process=Process.sequential,
            memory=self.shared_memory(),
            verbose=self.settings.crew_verbose,
        )

    def build_hierarchical_crew(self) -> Crew:
        return Crew(
            agents=self.build_agents(),
            tasks=self.build_tasks(),
            process=Process.hierarchical,
            memory=self.shared_memory(),
            manager_llm=self.manager_llm(),
            planning=self.settings.hierarchical_planning,
            verbose=self.settings.crew_verbose,
        )

    def config_dir(self) -> Path:
        return Path(self.base_directory) / "config"
