from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ActorRole = Literal["customer", "agent"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    user_id: str
    session_id: str
    role: ActorRole = "customer"
    request_id: str = ""
    display_name: str | None = None

