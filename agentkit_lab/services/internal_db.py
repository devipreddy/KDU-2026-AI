"""Mock internal analytics database service."""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import InternalDatabaseUnavailableError


@dataclass(slots=True)
class InternalDatabaseService:
    should_fail: bool = True
    active_user_count: int = 1287

    def count_active_users(self) -> int:
        if self.should_fail:
            raise InternalDatabaseUnavailableError(
                "HTTP 500 from query_internal_database"
            )
        return self.active_user_count
