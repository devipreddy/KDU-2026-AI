"""Finance domain service."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import BankingUpdateFact, FinanceRequest


@dataclass(slots=True)
class FinanceService:
    salaries: dict[str, int] = field(
        default_factory=lambda: {"john": 180000, "sarah": 155000}
    )
    bank_records: dict[str, BankingUpdateFact] = field(default_factory=dict)

    def get_salary(self, employee_name: str) -> str:
        salary = self.salaries[employee_name.lower()]
        return f"{employee_name}'s salary is ${salary:,} per year."

    def update_banking_details(self, request: FinanceRequest) -> str:
        employee_name = (request.employee_name or "unknown").lower()
        current = self.bank_records.setdefault(employee_name, BankingUpdateFact())
        if request.routing_number:
            current.routing_number = request.routing_number
        if request.account_number:
            current.account_number = request.account_number
        if request.account_holder_name:
            current.account_holder_name = request.account_holder_name

        missing = []
        if not current.routing_number:
            missing.append("routing_number")
        if not current.account_number:
            missing.append("account_number")
        if not current.account_holder_name:
            missing.append("account_holder_name")

        if missing:
            joined = ", ".join(missing)
            return (
                "Banking details were stored partially, but the update is not ready to apply. "
                f"Missing fields: {joined}."
            )

        return "Banking details are complete and ready for downstream approval."
