from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolExecutionError(RuntimeError):
    pass


class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            raise ToolExecutionError(f"Unknown tool: {name}")
        result = await tool.run(arguments)
        return {"tool_name": name, **result}
