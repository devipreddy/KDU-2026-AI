from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..schemas import Currency
from ..tracing import traceable


class FXRateError(RuntimeError):
    """Raised when a currency conversion rate is unavailable."""


class FXProvider(Protocol):
    provider_name: str

    def get_rate(self, from_currency: Currency, to_currency: Currency) -> float:
        """Return the conversion rate."""

    def convert(self, amount: float, from_currency: Currency, to_currency: Currency) -> float:
        """Convert an amount between currencies."""


@dataclass
class StaticFXProvider:
    provider_name: str = "static-fx"
    rates: dict[tuple[Currency, Currency], float] = field(
        default_factory=lambda: {
            ("USD", "INR"): 83.00,
            ("EUR", "INR"): 90.00,
            ("EUR", "USD"): 1.08,
        }
    )

    @traceable(name="fx_get_rate")
    def get_rate(self, from_currency: Currency, to_currency: Currency) -> float:
        if from_currency == to_currency:
            return 1.0
        direct = self.rates.get((from_currency, to_currency))
        if direct is not None:
            return direct
        inverse = self.rates.get((to_currency, from_currency))
        if inverse is not None:
            return round(1 / inverse, 8)

        if from_currency != "USD" and to_currency != "USD":
            via_usd = self.get_rate(from_currency, "USD") * self.get_rate("USD", to_currency)
            return round(via_usd, 8)

        raise FXRateError(f"FX rate unavailable for {from_currency} -> {to_currency}.")

    @traceable(name="fx_convert")
    def convert(self, amount: float, from_currency: Currency, to_currency: Currency) -> float:
        return round(amount * self.get_rate(from_currency, to_currency), 4)

