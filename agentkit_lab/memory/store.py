"""Simple JSON-backed persistence for messages and case facts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..schemas import CaseFacts, ConversationMessage, WorkingMemoryState


class TranscriptStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    def append(self, session_id: str, message: ConversationMessage) -> None:
        path = self._path(session_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(message)) + "\n")

    def load(self, session_id: str) -> list[ConversationMessage]:
        path = self._path(session_id)
        if not path.exists():
            return []
        messages: list[ConversationMessage] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                messages.append(ConversationMessage(**payload))
        return messages


class CaseFactsStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def load(self, session_id: str) -> CaseFacts:
        path = self._path(session_id)
        if not path.exists():
            return CaseFacts(session_id=session_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        transactions = payload.get("transactions", {})
        payload["transactions"] = {
            key: transactions[key] for key in transactions
        }
        facts = CaseFacts(session_id=session_id)
        facts.summary = payload.get("summary", "")
        facts.numerical_data = payload.get("numerical_data", {})
        financials = payload.get("financials", {})
        facts.financials.captured_amounts = financials.get("captured_amounts", [])
        facts.financials.currencies = financials.get("currencies", [])
        facts.financials.latest_balance = financials.get("latest_balance")
        facts.recent_relevant_messages = payload.get("recent_relevant_messages", [])
        facts.requires_user_input = payload.get("requires_user_input", False)
        facts.missing_fields = payload.get("missing_fields", [])
        facts.last_compacted_at = payload.get("last_compacted_at")
        facts.final_decisions = payload.get("final_decisions", [])
        banking = payload.get("banking_update", {})
        facts.banking_update.routing_number = banking.get("routing_number")
        facts.banking_update.account_number = banking.get("account_number")
        facts.banking_update.account_holder_name = banking.get("account_holder_name")
        facts.banking_update.card_last4 = banking.get("card_last4")
        facts.banking_update.cvv = banking.get("cvv")
        facts.banking_update.update_type = banking.get("update_type", "bank_account")
        for key, order in payload.get("orders", {}).items():
            from ..schemas import OrderFact

            facts.orders[key] = OrderFact(**order)
        for key, transaction in payload.get("transactions", {}).items():
            from ..schemas import TransactionFact

            facts.transactions[key] = TransactionFact(**transaction)
        return facts

    def save(self, facts: CaseFacts) -> None:
        path = self._path(facts.session_id)
        path.write_text(json.dumps(asdict(facts), indent=2), encoding="utf-8")


class WorkingMemoryStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def load(self, session_id: str) -> WorkingMemoryState:
        path = self._path(session_id)
        if not path.exists():
            return WorkingMemoryState(session_id=session_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return WorkingMemoryState(
            session_id=session_id,
            current_task=payload.get("current_task"),
            active_entities=payload.get("active_entities", {}),
            incremental_summary=payload.get("incremental_summary", ""),
            recent_decisions=payload.get("recent_decisions", []),
            pending_questions=payload.get("pending_questions", []),
            last_compacted_message_index=payload.get("last_compacted_message_index", 0),
            extraction_strategy=payload.get(
                "extraction_strategy",
                "cheap_deterministic_extractor",
            ),
            reasoning_strategy=payload.get("reasoning_strategy", "reasoning_agent"),
        )

    def save(self, working_memory: WorkingMemoryState) -> None:
        path = self._path(working_memory.session_id)
        path.write_text(json.dumps(asdict(working_memory), indent=2), encoding="utf-8")
