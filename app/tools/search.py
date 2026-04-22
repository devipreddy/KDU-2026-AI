from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings
from app.resilience import AsyncCircuitBreaker, CircuitBreakerOpenError
from app.tools.base import BaseTool, ToolExecutionError


class SearchTool(BaseTool):
    name = "search"
    description = "Search the web for recent or external information using Serper."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Web search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of organic results to include.",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient,
        circuit_breaker: AsyncCircuitBreaker,
    ) -> None:
        self.settings = settings
        self.http_client = http_client
        self.circuit_breaker = circuit_breaker

    async def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        max_results = max(1, min(int(arguments.get("max_results", 5)), 10))
        if not query:
            raise ToolExecutionError("Search query cannot be empty.")
        if not self.settings.serper_api_key:
            raise ToolExecutionError("SERPER_API_KEY is not configured.")

        try:
            return await self.circuit_breaker.call(
                self._perform_search,
                query,
                max_results,
            )
        except CircuitBreakerOpenError as exc:
            raise ToolExecutionError(
                f"Search is temporarily unavailable because the circuit breaker is open. {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"Search request failed: {exc}") from exc

    async def _perform_search(self, query: str, max_results: int) -> dict[str, Any]:
        response = await self.http_client.post(
            f"{self.settings.serper_base_url.rstrip('/')}/search",
            headers={"X-API-KEY": self.settings.serper_api_key},
            json={"q": query, "num": max_results},
        )
        response.raise_for_status()
        payload = response.json()

        organic = payload.get("organic", [])[:max_results]
        results = [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
            for item in organic
        ]
        answer_box = payload.get("answerBox") or payload.get("knowledgeGraph")
        news = payload.get("news", [])[:3]

        return {
            "status": "success",
            "query": query,
            "answer_box": answer_box,
            "results": results,
            "news": [
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "date": item.get("date"),
                }
                for item in news
            ],
        }
