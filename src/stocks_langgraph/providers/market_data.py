from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import quote
from urllib.request import urlopen

from ..config import Settings
from ..schemas import MarketQuote, ensure_supported_currency
from ..tracing import traceable


class MarketDataError(RuntimeError):
    """Raised when a quote cannot be retrieved."""


class MarketDataProvider(Protocol):
    provider_name: str
    supported_symbols: tuple[str, ...]

    def get_quote(self, symbol: str) -> MarketQuote:
        """Return a market quote for the requested symbol."""


@dataclass
class MockMarketDataProvider:
    provider_name: str = "mock"
    quotes: dict[str, tuple[float, str]] = field(
        default_factory=lambda: {
            "RELIANCE.NS": (2_950.50, "INR"),
            "INFY.NS": (1_512.75, "INR"),
            "SAP.DE": (181.20, "EUR"),
            "AAPL": (212.40, "USD"),
        }
    )

    @property
    def supported_symbols(self) -> tuple[str, ...]:
        return tuple(self.quotes.keys())

    @traceable(name="mock_market_data_quote")
    def get_quote(self, symbol: str) -> MarketQuote:
        normalized = symbol.upper()
        if normalized not in self.quotes:
            raise MarketDataError(f"No mock quote configured for symbol '{normalized}'.")
        price, currency = self.quotes[normalized]
        return MarketQuote(
            symbol=normalized,
            price_native=price,
            currency_native=ensure_supported_currency(currency),
            provider=self.provider_name,
        )


@dataclass
class YahooFinanceMarketDataProvider:
    provider_name: str = "yahoo"
    supported_symbols: tuple[str, ...] = ()

    @traceable(name="yahoo_finance_quote")
    def get_quote(self, symbol: str) -> MarketQuote:
        normalized = symbol.upper()
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={quote(normalized)}"
        try:
            with urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - requires external network.
            raise MarketDataError(f"Yahoo Finance request failed for '{normalized}': {exc}") from exc

        results = payload.get("quoteResponse", {}).get("result", [])
        if not results:
            raise MarketDataError(f"Yahoo Finance returned no result for '{normalized}'.")

        result = results[0]
        price = result.get("regularMarketPrice")
        currency = result.get("currency")
        if price is None or not currency:
            raise MarketDataError(f"Yahoo Finance response missing price/currency for '{normalized}'.")

        return MarketQuote(
            symbol=normalized,
            price_native=float(price),
            currency_native=ensure_supported_currency(currency),
            provider=self.provider_name,
        )


def build_market_data_provider(settings: Settings) -> MarketDataProvider:
    if settings.market_data_provider == "yahoo":
        return YahooFinanceMarketDataProvider()
    return MockMarketDataProvider()

