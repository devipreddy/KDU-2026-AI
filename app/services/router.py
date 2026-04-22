from __future__ import annotations

import re
from dataclasses import dataclass


MATH_ALLOWED_RE = re.compile(r"^[0-9A-Za-z\s\+\-\*/\^\(\)\.,%!]+$")
SEARCH_HINTS = (
    "search",
    "look up",
    "lookup",
    "find information",
    "find news",
    "latest",
    "current",
    "today",
    "news",
    "headline",
    "recent",
)
EXPLICIT_SEARCH_HINTS = (
    "search",
    "look up",
    "lookup",
    "find information",
    "find news",
    "news",
    "headline",
)
WEATHER_HINTS = ("weather", "forecast", "temperature", "rain", "humidity", "wind")
COMPLEXITY_HINTS = (
    "compare",
    "analyze",
    "analysis",
    "plan",
    "step by step",
    "explain why",
    "summarize",
    "pros and cons",
)
PLANNER_CONNECTORS = ("compare", "with", "and", "between", "vs", "versus")


@dataclass
class RouteDecision:
    route: str
    intent: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    confidence: float | None = None
    complexity: str | None = None


def sanitize_query(raw_query: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_query or "").strip()
    return cleaned


def classify_complexity(query: str) -> str:
    lowered = query.lower()
    if len(query) > 160 or lowered.count("?") > 1:
        return "high"
    if any(hint in lowered for hint in COMPLEXITY_HINTS):
        return "high"
    return "low"


def _looks_like_math(candidate: str) -> bool:
    stripped = candidate.strip().rstrip("?")
    if not stripped:
        return False
    if "__" in stripped or "[" in stripped or "]" in stripped:
        return False
    if not MATH_ALLOWED_RE.match(stripped):
        return False
    lowered = stripped.lower()
    if any(char.isdigit() for char in stripped):
        return True
    if any(operator in stripped for operator in ("+", "-", "*", "/", "^", "%", "(", ")")):
        return True
    return bool(
        re.search(r"\b(sin|cos|tan|sqrt|log|ln|pi|e|factorial|abs)\b", lowered)
    )


def extract_math_expression(query: str) -> str | None:
    stripped = query.strip().rstrip("?")
    lowered = stripped.lower()
    prefixes = ("calculate ", "compute ", "evaluate ", "solve ", "what is ", "what's ")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            candidate = stripped[len(prefix) :].strip(" :")
            return candidate if _looks_like_math(candidate) else None
    return stripped if _looks_like_math(stripped) else None


def extract_weather_location(query: str) -> tuple[str, str] | None:
    lowered = query.lower()
    if not any(hint in lowered for hint in WEATHER_HINTS):
        return None

    units = "fahrenheit" if "fahrenheit" in lowered else "celsius"
    match = re.search(
        r"\b(?:in|at|for)\b\s+([A-Za-z][A-Za-z\s,\.-]+)",
        query,
        re.IGNORECASE,
    )
    if match:
        location = match.group(1).strip(" ?.,")
        location = re.sub(
            r"\b(right now|now|today|tomorrow)\b",
            "",
            location,
            flags=re.IGNORECASE,
        ).strip(" ,")
        return (location, units) if location else None

    cleaned = re.sub(
        r"\b(weather|forecast|temperature|rain|humidity|wind|what's|what is|show me)\b",
        "",
        query,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(right now|now|today|tomorrow)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ?.,")
    return (cleaned, units) if cleaned else None


def should_search(query: str) -> bool:
    lowered = query.lower()
    has_weather_hint = any(hint in lowered for hint in WEATHER_HINTS)
    has_explicit_search_hint = any(hint in lowered for hint in EXPLICIT_SEARCH_HINTS)

    if has_weather_hint and not has_explicit_search_hint:
        return False

    if any(hint in lowered for hint in SEARCH_HINTS):
        return True
    # Route factual "who/what/when" queries with recency hints to web search.
    if re.match(r"^(who|what|when|where)\b", lowered) and any(
        hint in lowered for hint in ("latest", "current", "today", "recent")
    ):
        return True
    return False


def detect_tool_domains(query: str) -> set[str]:
    domains: set[str] = set()
    if extract_math_expression(query):
        domains.add("calculator")
    if extract_weather_location(query):
        domains.add("weather")
    if should_search(query):
        domains.add("search")
    return domains


def needs_llm_planner(query: str, complexity: str, domains: set[str]) -> bool:
    if len(domains) > 1:
        return True

    lowered = query.lower()
    has_planner_connector = any(token in lowered for token in PLANNER_CONNECTORS)

    # Weather comparisons usually imply multiple tool calls or reasoning steps,
    # even if the only explicit domain is weather.
    if complexity == "high" and "weather" in domains and has_planner_connector:
        return True

    return False


def decide_route(query: str) -> RouteDecision:
    complexity = classify_complexity(query)
    domains = detect_tool_domains(query)

    if needs_llm_planner(query, complexity, domains):
        return RouteDecision(
            route="llm",
            intent="multi_tool" if len(domains) > 1 else "reasoning",
            tool_name=None,
            tool_input=None,
            confidence=0.9 if len(domains) > 1 else 0.76,
            complexity=complexity,
        )

    expression = extract_math_expression(query)
    if expression:
        return RouteDecision(
            route="tool",
            intent="calculate",
            tool_name="calculator",
            tool_input={"expression": expression, "precision": 12},
            confidence=0.98,
            complexity=complexity,
        )

    weather = extract_weather_location(query)
    if weather:
        location, units = weather
        return RouteDecision(
            route="tool",
            intent="weather",
            tool_name="weather",
            tool_input={"location": location, "units": units},
            confidence=0.95,
            complexity=complexity,
        )

    if should_search(query):
        return RouteDecision(
            route="tool",
            intent="search",
            tool_name="search",
            tool_input={"query": query, "max_results": 5},
            confidence=0.82,
            complexity=complexity,
        )

    return RouteDecision(
        route="llm",
        intent="general",
        tool_name=None,
        tool_input=None,
        confidence=0.5,
        complexity=complexity,
    )
