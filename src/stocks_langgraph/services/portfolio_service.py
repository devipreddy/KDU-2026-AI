from __future__ import annotations

from dataclasses import dataclass

from ..schemas import DomainState, Holding, MarketQuote, Order, OrderIntent, Transaction, utc_now
from ..tracing import traceable


@dataclass
class PortfolioService:
    @traceable(name="portfolio_validate_order")
    def validate_order(self, domain: DomainState, intent: OrderIntent, quote: MarketQuote) -> str | None:
        if intent.quantity is None or intent.quantity <= 0:
            return "Trade quantity must be greater than zero."
        if quote.price_base is None or quote.base_currency is None:
            return "Quote must be normalized into the portfolio base currency before trading."

        notional = round(intent.quantity * quote.price_base, 4)
        if intent.action == "BUY" and notional > domain.portfolio.cash_balance_base:
            return "Insufficient cash balance for the requested BUY order."

        if intent.action == "SELL":
            holding = domain.portfolio.holdings.get(intent.symbol or "")
            if holding is None:
                return f"No holding found for symbol '{intent.symbol}'."
            if intent.quantity > holding.quantity:
                return f"Cannot sell {intent.quantity} shares of '{intent.symbol}'; only {holding.quantity} available."

        return None

    @traceable(name="portfolio_apply_order")
    def apply_order(self, domain: DomainState, intent: OrderIntent, quote: MarketQuote) -> DomainState:
        if intent.symbol is None or intent.quantity is None:
            raise ValueError("Intent is missing symbol or quantity.")
        if quote.price_base is None or quote.base_currency is None:
            raise ValueError("Quote must contain base currency pricing before execution.")

        quantity = intent.quantity
        symbol = intent.symbol
        notional = round(quantity * quote.price_base, 4)
        existing = domain.portfolio.holdings.get(symbol)

        if intent.action == "BUY":
            previous_qty = existing.quantity if existing else 0.0
            previous_cost = existing.average_cost_base if existing else 0.0
            total_cost = previous_qty * previous_cost + notional
            new_quantity = previous_qty + quantity
            average_cost = round(total_cost / new_quantity, 4)
            domain.portfolio.cash_balance_base = round(domain.portfolio.cash_balance_base - notional, 4)
            domain.portfolio.holdings[symbol] = Holding(
                symbol=symbol,
                quantity=new_quantity,
                average_cost_base=average_cost,
                price_native=quote.price_native,
                currency_native=quote.currency_native,
                price_base=quote.price_base,
                market_value_base=round(new_quantity * quote.price_base, 4),
                updated_at=utc_now(),
            )
        elif intent.action == "SELL":
            if existing is None:
                raise ValueError(f"No holding found for '{symbol}'.")
            remaining = round(existing.quantity - quantity, 4)
            domain.portfolio.cash_balance_base = round(domain.portfolio.cash_balance_base + notional, 4)
            if remaining <= 0:
                domain.portfolio.holdings.pop(symbol, None)
            else:
                domain.portfolio.holdings[symbol] = Holding(
                    symbol=symbol,
                    quantity=remaining,
                    average_cost_base=existing.average_cost_base,
                    price_native=quote.price_native,
                    currency_native=quote.currency_native,
                    price_base=quote.price_base,
                    market_value_base=round(remaining * quote.price_base, 4),
                    updated_at=utc_now(),
                )
        else:
            raise ValueError(f"Unsupported trade action '{intent.action}'.")

        order = Order(
            action=intent.action,
            symbol=symbol,
            quantity=quantity,
            status="EXECUTED",
            price_base=quote.price_base,
            notional_base=notional,
            currency_base=quote.base_currency,
        )
        transaction = Transaction(
            order_id=order.order_id,
            action=intent.action,
            symbol=symbol,
            quantity=quantity,
            amount_base=notional,
            currency_base=quote.base_currency,
        )

        domain.orders.append(order)
        domain.ledger.append(transaction)
        return self.recalculate(domain)

    @traceable(name="portfolio_recalculate")
    def recalculate(self, domain: DomainState) -> DomainState:
        total_market_value = 0.0
        for holding in domain.portfolio.holdings.values():
            holding.market_value_base = round(holding.quantity * holding.price_base, 4)
            holding.updated_at = utc_now()
            total_market_value += holding.market_value_base

        domain.portfolio.total_market_value_base = round(total_market_value, 4)
        domain.portfolio.total_equity_base = round(
            domain.portfolio.cash_balance_base + domain.portfolio.total_market_value_base,
            4,
        )
        return domain

    @traceable(name="portfolio_summary")
    def build_summary(self, domain: DomainState) -> str:
        portfolio = self.recalculate(domain).portfolio
        lines = [
            f"Base currency: {domain.user.base_currency}",
            f"Cash: {portfolio.cash_balance_base:.2f} {domain.user.base_currency}",
            f"Market value: {portfolio.total_market_value_base:.2f} {domain.user.base_currency}",
            f"Total equity: {portfolio.total_equity_base:.2f} {domain.user.base_currency}",
        ]

        if portfolio.holdings:
            holdings_text = ", ".join(
                f"{holding.symbol} x {holding.quantity} @ {holding.price_base:.2f} {domain.user.base_currency}"
                for holding in portfolio.holdings.values()
            )
            lines.append(f"Holdings: {holdings_text}")
        else:
            lines.append("Holdings: none")

        return " | ".join(lines)
