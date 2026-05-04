from __future__ import annotations

from pathlib import Path

import orjson

from app.core.models import LoggedEvent


class SessionReplayStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, event: LoggedEvent) -> None:
        path = self.root / f"{event.session_id}.ndjson"
        with path.open("ab") as handle:
            handle.write(orjson.dumps(event.model_dump(mode="json")))
            handle.write(b"\n")

    def read(self, session_id: str) -> list[dict]:
        path = self.root / f"{session_id}.ndjson"
        if not path.exists():
            return []
        return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
