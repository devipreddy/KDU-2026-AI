from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import RequestLogRecord


class SQLiteStateStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_log (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    category TEXT NOT NULL,
                    complexity TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    requested_model_alias TEXT NOT NULL,
                    used_model_alias TEXT,
                    model_id TEXT,
                    prompt_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    budget_action TEXT NOT NULL,
                    classification_confidence REAL NOT NULL DEFAULT 0,
                    response_preview TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_usage (
                    prompt_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    avg_estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    last_used_at TEXT,
                    PRIMARY KEY (prompt_id, version)
                )
                """
            )

    def record_request(self, record: RequestLogRecord) -> None:
        timestamp = record.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO request_log (
                    request_id, timestamp, query, category, complexity, response_type,
                    requested_model_alias, used_model_alias, model_id, prompt_id,
                    prompt_version, prompt_tokens, completion_tokens, estimated_cost_usd,
                    budget_action, classification_confidence, response_preview
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    timestamp,
                    record.query,
                    record.category,
                    record.complexity,
                    record.response_type,
                    record.requested_model_alias,
                    record.used_model_alias,
                    record.model_id,
                    record.prompt_id,
                    record.prompt_version,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.estimated_cost_usd,
                    record.budget_action,
                    record.classification_confidence,
                    record.response_preview,
                ),
            )

    def record_prompt_use(self, prompt_id: str, version: str, estimated_cost_usd: float, success: bool = True) -> None:
        timestamp = datetime.utcnow().isoformat()
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT usage_count, success_count, avg_estimated_cost_usd FROM prompt_usage WHERE prompt_id = ? AND version = ?",
                (prompt_id, version),
            ).fetchone()

            if existing is None:
                usage_count = 1
                success_count = 1 if success else 0
                avg_cost = estimated_cost_usd
            else:
                usage_count = int(existing["usage_count"]) + 1
                success_count = int(existing["success_count"]) + (1 if success else 0)
                old_avg = float(existing["avg_estimated_cost_usd"])
                avg_cost = ((old_avg * (usage_count - 1)) + estimated_cost_usd) / usage_count

            conn.execute(
                """
                INSERT OR REPLACE INTO prompt_usage (
                    prompt_id, version, usage_count, success_count, avg_estimated_cost_usd, last_used_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (prompt_id, version, usage_count, success_count, avg_cost, timestamp),
            )

    def get_spend_between(self, start_iso: str, end_iso: str) -> float:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(estimated_cost_usd), 0) AS spend
                FROM request_log
                WHERE timestamp >= ? AND timestamp < ?
                """,
                (start_iso, end_iso),
            ).fetchone()
        return float(row["spend"]) if row else 0.0
