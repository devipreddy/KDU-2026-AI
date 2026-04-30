"""Regex-based fact extraction for durable case memory."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import BankingUpdateFact, TransactionFact

ORDER_ID_RE = re.compile(r"\b(?:order|ord)[\s#:-]*([A-Z0-9-]{3,})\b", re.IGNORECASE)
TRANSACTION_ID_RE = re.compile(
    r"\b(?:transaction|txn)[\s#:-]*([A-Z0-9-]{3,})\b",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(r"(?:(USD)\s*|\$)\s*([0-9]+(?:\.[0-9]{2})?)|\b([0-9]+\.[0-9]{2})\b")
ROUTING_RE = re.compile(r"\b\d{9}\b")
ACCOUNT_RE = re.compile(r"\baccount(?: number)?[\s:=-]*([0-9]{4,17})\b", re.IGNORECASE)
CVV_RE = re.compile(r"\bcvv[\s:=-]*([0-9]{3,4})\b", re.IGNORECASE)
CARD_LAST4_RE = re.compile(r"\b(?:last\s*4|ending\s*in)[\s:=-]*([0-9]{4})\b", re.IGNORECASE)
CURRENCY_HINT_RE = re.compile(r"\b(USD|INR|EUR|GBP)\b", re.IGNORECASE)
BALANCE_RE = re.compile(
    r"\bbalance(?:\s+is|:)?\s*(?:USD|INR|EUR|GBP|\$|₹)?\s*([0-9]+(?:\.[0-9]{2})?)",
    re.IGNORECASE,
)
STATUS_KEYWORDS = {
    "settled": "settled",
    "pending": "pending",
    "failed": "failed",
    "completed": "completed",
    "refunded": "refunded",
    "review": "in_review",
}


@dataclass(slots=True)
class ExtractionResult:
    numerical_data: dict[str, list[str]] = field(default_factory=dict)
    transactions: list[TransactionFact] = field(default_factory=list)
    banking_update: BankingUpdateFact | None = None
    currencies: list[str] = field(default_factory=list)
    latest_balance: str | None = None


class FactExtractor:
    def _status_from_text(self, text: str) -> str | None:
        lowered = text.lower()
        for keyword, normalized in STATUS_KEYWORDS.items():
            if keyword in lowered:
                return normalized
        return None

    def _extract_amounts(self, text: str) -> list[str]:
        amounts: list[str] = []
        for match in AMOUNT_RE.finditer(text):
            amount = match.group(2) or match.group(3)
            if amount is not None:
                amounts.append(amount)
        return amounts

    def extract(self, text: str) -> ExtractionResult:
        result = ExtractionResult()
        amounts = self._extract_amounts(text)
        if amounts:
            result.numerical_data["amounts"] = amounts

        currencies = [match.group(1).upper() for match in CURRENCY_HINT_RE.finditer(text)]
        if "$" in text and "USD" not in currencies:
            currencies.append("USD")
        if "₹" in text and "INR" not in currencies:
            currencies.append("INR")
        result.currencies = sorted(set(currencies))

        balance_match = BALANCE_RE.search(text)
        if balance_match:
            result.latest_balance = balance_match.group(1)

        routing_numbers = ROUTING_RE.findall(text)
        if routing_numbers:
            result.numerical_data["routing_numbers"] = routing_numbers

        order_ids = [match.group(1) for match in ORDER_ID_RE.finditer(text)]
        transaction_ids = [match.group(1) for match in TRANSACTION_ID_RE.finditer(text)]
        if order_ids:
            result.numerical_data["order_ids"] = order_ids
        if transaction_ids:
            result.numerical_data["transaction_ids"] = transaction_ids

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        extracted_any_transaction = False
        for sentence in sentences:
            sentence_order_ids = [match.group(1) for match in ORDER_ID_RE.finditer(sentence)]
            sentence_txn_ids = [match.group(1) for match in TRANSACTION_ID_RE.finditer(sentence)]
            sentence_amounts = self._extract_amounts(sentence)
            if not sentence_order_ids and not sentence_txn_ids:
                continue

            extracted_any_transaction = True
            status = self._status_from_text(sentence)
            primary_amount = sentence_amounts[0] if sentence_amounts else None
            primary_txn_id = sentence_txn_ids[0] if sentence_txn_ids else None

            for order_id in sentence_order_ids:
                result.transactions.append(
                    TransactionFact(
                        fact_id=f"order-{order_id.lower()}",
                        order_id=order_id,
                        transaction_id=primary_txn_id,
                        amount=primary_amount,
                        currency="USD" if primary_amount else None,
                        status=status,
                        raw_excerpt=sentence[:280],
                    )
                )

            for transaction_id in sentence_txn_ids:
                result.transactions.append(
                    TransactionFact(
                        fact_id=f"txn-{transaction_id.lower()}",
                        order_id=sentence_order_ids[0] if sentence_order_ids else None,
                        transaction_id=transaction_id,
                        amount=primary_amount,
                        currency="USD" if primary_amount else None,
                        status=status,
                        raw_excerpt=sentence[:280],
                    )
                )

        if transaction_ids and not extracted_any_transaction:
            for index, transaction_id in enumerate(transaction_ids):
                amount = amounts[index] if index < len(amounts) else None
                result.transactions.append(
                    TransactionFact(
                        fact_id=f"txn-{transaction_id.lower()}",
                        transaction_id=transaction_id,
                        amount=amount,
                        currency="USD" if amount else None,
                        raw_excerpt=text[:280],
                    )
                )

        banking_update = BankingUpdateFact()
        match = ACCOUNT_RE.search(text)
        if match:
            banking_update.account_number = match.group(1)

        if routing_numbers:
            banking_update.routing_number = routing_numbers[0]

        match = CVV_RE.search(text)
        if match:
            banking_update.update_type = "payment_card"
            banking_update.cvv = match.group(1)

        match = CARD_LAST4_RE.search(text)
        if match:
            banking_update.update_type = "payment_card"
            banking_update.card_last4 = match.group(1)

        if (
            banking_update.routing_number
            or banking_update.account_number
            or banking_update.card_last4
            or banking_update.cvv
        ):
            result.banking_update = banking_update

        return result
