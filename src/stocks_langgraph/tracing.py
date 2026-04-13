from __future__ import annotations

from collections.abc import Callable
from typing import Any


try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - fallback for environments without langsmith.

    def traceable(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

