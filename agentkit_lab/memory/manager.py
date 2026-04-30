"""Structured memory manager with compaction and fact preservation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..schemas import (
    CaseFacts,
    ConversationMessage,
    MemoryCompactionReport,
    OrderFact,
    WorkingMemoryState,
    to_plain_data,
    utc_now_iso,
)
from .fact_extractor import FactExtractor
from .store import CaseFactsStore, TranscriptStore, WorkingMemoryStore


LOW_SIGNAL_MESSAGES = {"ok", "okay", "cool", "thanks", "thank you", "noted"}


class StructuredMemoryManager:
    def __init__(self, base_dir: Path, compaction_char_threshold: int, max_recent_messages: int) -> None:
        self.transcript_store = TranscriptStore(base_dir / "transcripts")
        self.working_memory_store = WorkingMemoryStore(base_dir / "working_memory")
        self.case_facts_store = CaseFactsStore(base_dir / "case_facts")
        self.extractor = FactExtractor()
        self.compaction_char_threshold = compaction_char_threshold
        self.max_recent_messages = max_recent_messages

    def ingest_message(self, session_id: str, role: str, content: str) -> CaseFacts:
        message = ConversationMessage(role=role, content=content)
        self.transcript_store.append(session_id, message)
        facts = self.case_facts_store.load(session_id)
        working_memory = self.working_memory_store.load(session_id)
        extraction = self.extractor.extract(content)

        for key, values in extraction.numerical_data.items():
            existing = facts.numerical_data.setdefault(key, [])
            for value in values:
                if value not in existing:
                    existing.append(value)

        for amount in extraction.numerical_data.get("amounts", []):
            if amount not in facts.financials.captured_amounts:
                facts.financials.captured_amounts.append(amount)
        for currency in extraction.currencies:
            if currency not in facts.financials.currencies:
                facts.financials.currencies.append(currency)
        if extraction.latest_balance is not None:
            facts.financials.latest_balance = extraction.latest_balance

        for transaction in extraction.transactions:
            facts.transactions[transaction.fact_id] = transaction
            if transaction.order_id:
                facts.orders[transaction.order_id] = OrderFact(
                    id=transaction.order_id,
                    amount=transaction.amount,
                    currency=transaction.currency,
                    transaction_id=transaction.transaction_id,
                    status=transaction.status,
                    raw_excerpt=transaction.raw_excerpt,
                )

        if extraction.banking_update is not None:
            current = facts.banking_update
            if extraction.banking_update.routing_number:
                current.routing_number = extraction.banking_update.routing_number
            if extraction.banking_update.account_number:
                current.account_number = extraction.banking_update.account_number
            if extraction.banking_update.card_last4:
                current.card_last4 = extraction.banking_update.card_last4
            if extraction.banking_update.cvv:
                current.cvv = extraction.banking_update.cvv
            current.update_type = extraction.banking_update.update_type

        self._refresh_flags(facts)
        self._update_working_memory(
            working_memory,
            role=role,
            content=content,
            facts=facts,
            extraction=extraction,
        )
        self._maybe_compact(session_id, facts, working_memory)
        self.case_facts_store.save(facts)
        self.working_memory_store.save(working_memory)
        return facts

    def get_case_facts(self, session_id: str) -> CaseFacts:
        return self.case_facts_store.load(session_id)

    def get_working_memory(self, session_id: str) -> WorkingMemoryState:
        return self.working_memory_store.load(session_id)

    def build_compaction_report(self, session_id: str) -> MemoryCompactionReport:
        facts = self.case_facts_store.load(session_id)
        working_memory = self.working_memory_store.load(session_id)
        return MemoryCompactionReport(
            session_id=session_id,
            retained_message_count=len(facts.recent_relevant_messages),
            summary=facts.summary,
            numerical_keys=sorted(facts.numerical_data.keys()),
            transaction_ids=sorted(facts.transactions.keys()),
            requires_user_input=facts.requires_user_input,
            missing_fields=list(facts.missing_fields),
            case_facts=to_plain_data(facts),
            working_memory=to_plain_data(working_memory),
            short_term_window=list(facts.recent_relevant_messages),
        )

    def _refresh_flags(self, facts: CaseFacts) -> None:
        missing_fields: list[str] = []
        update = facts.banking_update
        if update.update_type == "payment_card":
            if not update.card_last4:
                missing_fields.append("card_last4")
            if not update.cvv:
                missing_fields.append("cvv")
        else:
            if update.routing_number and not update.account_number:
                missing_fields.append("account_number")
            if update.routing_number and not update.account_holder_name:
                missing_fields.append("account_holder_name")

        facts.missing_fields = missing_fields
        facts.requires_user_input = bool(missing_fields)

    def _extract_final_decision(self, role: str, content: str) -> str | None:
        if role != "assistant":
            return None
        lowered = content.lower()
        if any(
            keyword in lowered
            for keyword in ("decision", "approved", "declined", "denied", "ready to apply")
        ):
            return content.strip()[:280]
        return None

    def _update_working_memory(
        self,
        working_memory: WorkingMemoryState,
        *,
        role: str,
        content: str,
        facts: CaseFacts,
        extraction,
    ) -> None:
        if extraction.transactions:
            working_memory.current_task = "transaction_review"
        if extraction.banking_update is not None:
            working_memory.current_task = (
                "payment_card_update"
                if extraction.banking_update.update_type == "payment_card"
                else "bank_account_update"
            )
        if facts.requires_user_input:
            working_memory.current_task = "collect_required_fields"

        entity_updates: dict[str, str] = {}
        if extraction.numerical_data.get("order_ids"):
            entity_updates["last_order_id"] = extraction.numerical_data["order_ids"][-1]
        if extraction.numerical_data.get("transaction_ids"):
            entity_updates["last_transaction_id"] = extraction.numerical_data["transaction_ids"][-1]
        if facts.banking_update.routing_number:
            entity_updates["routing_number"] = facts.banking_update.routing_number
        if facts.banking_update.card_last4:
            entity_updates["card_last4"] = facts.banking_update.card_last4
        if facts.financials.latest_balance:
            entity_updates["latest_balance"] = facts.financials.latest_balance
        working_memory.active_entities.update(entity_updates)
        working_memory.pending_questions = list(facts.missing_fields)

        final_decision = self._extract_final_decision(role, content)
        if final_decision is not None:
            facts.final_decisions = [final_decision]
            working_memory.recent_decisions = [final_decision]

    def _summarize_delta(
        self,
        delta_messages: list[ConversationMessage],
        facts: CaseFacts,
        working_memory: WorkingMemoryState,
    ) -> str:
        if not delta_messages:
            return ""

        delta_text = " ".join(message.content for message in delta_messages)
        delta_extraction = self.extractor.extract(delta_text)
        summary_parts: list[str] = []
        if delta_extraction.transactions:
            order_count = len({item.order_id for item in delta_extraction.transactions if item.order_id})
            txn_count = len({item.transaction_id for item in delta_extraction.transactions if item.transaction_id})
            summary_parts.append(
                f"Delta captured {order_count} orders and {txn_count} transactions."
            )
        if delta_extraction.numerical_data.get("amounts"):
            summary_parts.append(
                f"New exact amounts: {', '.join(delta_extraction.numerical_data['amounts'][:5])}."
            )
        if delta_extraction.numerical_data.get("order_ids"):
            summary_parts.append(
                f"New order IDs: {', '.join(delta_extraction.numerical_data['order_ids'][:5])}."
            )
        if delta_extraction.latest_balance is not None:
            summary_parts.append(
                f"Latest balance stored exactly as {delta_extraction.latest_balance}."
            )
        if facts.requires_user_input:
            summary_parts.append(
                f"Pending required fields: {', '.join(facts.missing_fields)}."
            )
        if facts.final_decisions:
            summary_parts.append(f"Latest final decision: {facts.final_decisions[-1]}.")
        if not summary_parts:
            summary_parts.append(
                f"Compacted {len(delta_messages)} relevant messages for active task {working_memory.current_task or 'general_follow_up'}."
            )
        return " ".join(summary_parts)

    def _maybe_compact(
        self,
        session_id: str,
        facts: CaseFacts,
        working_memory: WorkingMemoryState,
    ) -> None:
        messages = self.transcript_store.load(session_id)
        relevant_messages = [
            message.content
            for message in messages
            if message.content.strip().lower() not in LOW_SIGNAL_MESSAGES
        ]
        facts.recent_relevant_messages = relevant_messages[-self.max_recent_messages :]
        total_chars = sum(len(message.content) for message in messages)
        if total_chars < self.compaction_char_threshold:
            if not facts.summary and facts.transactions:
                facts.summary = (
                    f"Tracked {len(facts.transactions)} transaction facts without requiring full compaction yet."
                )
            return

        start_index = min(working_memory.last_compacted_message_index, len(messages))
        delta_messages = [
            message
            for message in messages[start_index:]
            if message.content.strip().lower() not in LOW_SIGNAL_MESSAGES
        ]
        if delta_messages:
            delta_summary = self._summarize_delta(delta_messages, facts, working_memory)
            if delta_summary:
                if working_memory.incremental_summary:
                    working_memory.incremental_summary += " " + delta_summary
                else:
                    working_memory.incremental_summary = delta_summary
                facts.summary = working_memory.incremental_summary

        working_memory.last_compacted_message_index = len(messages)
        facts.last_compacted_at = utc_now_iso()

    def clone_facts(self, session_id: str) -> CaseFacts:
        return replace(self.case_facts_store.load(session_id))

    def build_sdk_session(self, session_id: str, *, enable_openai_compaction: bool = True):
        try:
            from agents import SQLiteSession
        except ImportError:
            return None

        session_db_path = self.transcript_store.base_dir.parent / "sessions.sqlite3"
        underlying = SQLiteSession(session_id, str(session_db_path))
        if not enable_openai_compaction:
            return underlying

        try:
            from agents.memory import OpenAIResponsesCompactionSession
        except ImportError:
            return underlying

        return OpenAIResponsesCompactionSession(
            session_id=session_id,
            underlying_session=underlying,
            should_trigger_compaction=lambda ctx: len(
                getattr(ctx, "compaction_candidate_items", [])
            )
            >= self.max_recent_messages,
        )
