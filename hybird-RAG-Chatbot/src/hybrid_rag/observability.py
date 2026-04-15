from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import utc_timestamp, write_json_file


class TraceStore:
    def __init__(self, trace_dir: str) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def write_trace(self, session_id: str, payload: dict[str, Any]) -> str:
        timestamp = utc_timestamp().replace(":", "-")
        target = self.trace_dir / f"{session_id}_{timestamp}.json"
        write_json_file(str(target), payload)
        return str(target)
