"""HR domain service."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class HRService:
    pto_balances: dict[str, int] = field(default_factory=lambda: {"john": 12, "sarah": 18})

    def get_pto_balance(self, employee_name: str) -> str:
        balance = self.pto_balances[employee_name.lower()]
        return f"{employee_name} has {balance} days of PTO remaining."
