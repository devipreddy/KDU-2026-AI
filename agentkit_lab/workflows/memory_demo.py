"""Phase 4 memory compaction and case facts demo."""

from __future__ import annotations

from ..memory.manager import StructuredMemoryManager
from ..schemas import MemoryCompactionReport
from ..settings import AppSettings


def _build_long_transaction_document() -> str:
    filler_sentence = (
        "The customer reviewed the timeline, verified the shipment note, and confirmed the status update. "
    )
    filler_word_count = len(filler_sentence.split())
    target_word_count = 5000
    critical = (
        "Order ORD-8472 was charged $412.55 and transaction TXN-77A1 settled successfully. "
        "Order ORD-9321 was charged $89.99 and transaction TXN-11C9 is pending review. "
        "The customer requested a card update for last 4 5512 but did not provide CVV. "
    )
    critical_word_count = len(critical.split())
    filler_repetitions = max(
        1,
        (target_word_count - critical_word_count) // filler_word_count,
    )
    leading_repetitions = filler_repetitions // 2
    trailing_repetitions = filler_repetitions - leading_repetitions
    return (filler_sentence * leading_repetitions) + critical + (filler_sentence * trailing_repetitions)


def run_memory_demo(
    settings: AppSettings,
    *,
    session_id: str = "memory-demo",
) -> MemoryCompactionReport:
    memory_manager = StructuredMemoryManager(
        base_dir=settings.data_dir,
        compaction_char_threshold=settings.compaction_char_threshold,
        max_recent_messages=settings.max_recent_messages,
    )
    memory_manager.ingest_message(session_id, "user", _build_long_transaction_document())
    memory_manager.ingest_message(session_id, "user", "okay")
    memory_manager.ingest_message(session_id, "assistant", "cool")
    return memory_manager.build_compaction_report(session_id)
