from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path

import aiosqlite
from chatkit.store import AttachmentStore, NotFoundError, Store
from chatkit.types import (
    Attachment,
    AttachmentCreateParams,
    FileAttachment,
    Page,
    ThreadItem,
    ThreadMetadata,
)
from pydantic import TypeAdapter

from .context import RequestContext


THREAD_ADAPTER = TypeAdapter(ThreadMetadata)
THREAD_ITEM_ADAPTER = TypeAdapter(ThreadItem)
ATTACHMENT_ADAPTER = TypeAdapter(Attachment)


class SQLiteChatStore(Store[RequestContext], AttachmentStore[RequestContext]):
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as db:
            await db.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS thread_items (
                    sort_index INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    thread_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS action_receipts (
                    receipt_key TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            await db.commit()

    async def load_thread(self, thread_id: str, context: RequestContext) -> ThreadMetadata:
        row = await self._load_thread_row(thread_id, context)
        return THREAD_ADAPTER.validate_json(row["payload_json"])

    async def save_thread(self, thread: ThreadMetadata, context: RequestContext) -> None:
        existing_owner = await self._load_thread_owner(thread.id)
        owner_user_id = existing_owner or context.user_id
        if existing_owner is not None:
            self._assert_owner(existing_owner, context)

        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO threads (id, owner_user_id, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (thread.id, owner_user_id, thread.model_dump_json()),
            )
            await db.commit()

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: RequestContext,
    ) -> Page[ThreadItem]:
        await self._load_thread_row(thread_id, context)

        comparator = ">"
        sort_direction = "ASC"
        anchor_clause = ""
        anchor_params: list[object] = []

        if order == "desc":
            comparator = "<"
            sort_direction = "DESC"

        if after:
            anchor_sort_index = await self._lookup_item_sort_index(thread_id, after)
            anchor_clause = f"AND sort_index {comparator} ?"
            anchor_params.append(anchor_sort_index)

        query = f"""
            SELECT id, payload_json
            FROM thread_items
            WHERE thread_id = ?
            {anchor_clause}
            ORDER BY sort_index {sort_direction}
            LIMIT ?
        """

        async with self._connect() as db:
            cursor = await db.execute(query, [thread_id, *anchor_params, limit + 1])
            rows = await cursor.fetchall()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        items = [
            THREAD_ITEM_ADAPTER.validate_json(row["payload_json"]) for row in page_rows
        ]
        after_cursor = page_rows[-1]["id"] if has_more and page_rows else None
        return Page(data=items, has_more=has_more, after=after_cursor)

    async def save_attachment(
        self,
        attachment: Attachment,
        context: RequestContext,
    ) -> None:
        owner_user_id = await self._attachment_owner(attachment.id) or context.user_id
        if owner_user_id != context.user_id and context.role != "agent":
            raise NotFoundError(f"Attachment {attachment.id} was not found")

        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO attachments (id, owner_user_id, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (attachment.id, owner_user_id, ATTACHMENT_ADAPTER.dump_json(attachment)),
            )
            await db.commit()

    async def load_attachment(
        self,
        attachment_id: str,
        context: RequestContext,
    ) -> Attachment:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT owner_user_id, payload_json FROM attachments WHERE id = ?",
                (attachment_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            raise NotFoundError(f"Attachment {attachment_id} was not found")
        self._assert_owner(row["owner_user_id"], context)
        return ATTACHMENT_ADAPTER.validate_json(row["payload_json"])

    async def delete_attachment(
        self,
        attachment_id: str,
        context: RequestContext,
    ) -> None:
        owner = await self._attachment_owner(attachment_id)
        if owner is None:
            return
        self._assert_owner(owner, context)
        async with self._connect() as db:
            await db.execute("DELETE FROM attachments WHERE id = ?", (attachment_id,))
            await db.commit()

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: RequestContext,
    ) -> Page[ThreadMetadata]:
        comparator = ">"
        sort_direction = "ASC"
        anchor_clause = ""
        anchor_params: list[object] = []

        if order == "desc":
            comparator = "<"
            sort_direction = "DESC"

        if after:
            anchor_sort_index = await self._lookup_thread_sort_index(after)
            anchor_clause = f"AND rowid {comparator} ?"
            anchor_params.append(anchor_sort_index)

        filter_clause = ""
        filter_params: list[object] = []
        if context.role != "agent":
            filter_clause = "AND owner_user_id = ?"
            filter_params.append(context.user_id)

        query = f"""
            SELECT id, payload_json
            FROM threads
            WHERE 1 = 1
            {filter_clause}
            {anchor_clause}
            ORDER BY rowid {sort_direction}
            LIMIT ?
        """

        async with self._connect() as db:
            cursor = await db.execute(
                query,
                [*filter_params, *anchor_params, limit + 1],
            )
            rows = await cursor.fetchall()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        threads = [THREAD_ADAPTER.validate_json(row["payload_json"]) for row in page_rows]
        after_cursor = page_rows[-1]["id"] if has_more and page_rows else None
        return Page(data=threads, has_more=has_more, after=after_cursor)

    async def add_thread_item(
        self,
        thread_id: str,
        item: ThreadItem,
        context: RequestContext,
    ) -> None:
        row = await self._load_thread_row(thread_id, context)
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO thread_items (id, thread_id, owner_user_id, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (item.id, thread_id, row["owner_user_id"], THREAD_ITEM_ADAPTER.dump_json(item)),
            )
            await db.commit()

    async def save_item(
        self,
        thread_id: str,
        item: ThreadItem,
        context: RequestContext,
    ) -> None:
        row = await self._load_thread_row(thread_id, context)
        async with self._connect() as db:
            await db.execute(
                """
                UPDATE thread_items
                SET payload_json = ?
                WHERE id = ? AND thread_id = ? AND owner_user_id = ?
                """,
                (
                    THREAD_ITEM_ADAPTER.dump_json(item),
                    item.id,
                    thread_id,
                    row["owner_user_id"],
                ),
            )
            await db.commit()

    async def load_item(
        self,
        thread_id: str,
        item_id: str,
        context: RequestContext,
    ) -> ThreadItem:
        await self._load_thread_row(thread_id, context)
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT payload_json
                FROM thread_items
                WHERE id = ? AND thread_id = ?
                """,
                (item_id, thread_id),
            )
            row = await cursor.fetchone()

        if row is None:
            raise NotFoundError(f"Item {item_id} was not found")
        return THREAD_ITEM_ADAPTER.validate_json(row["payload_json"])

    async def delete_thread(self, thread_id: str, context: RequestContext) -> None:
        await self._load_thread_row(thread_id, context)
        async with self._connect() as db:
            await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            await db.commit()

    async def delete_thread_item(
        self,
        thread_id: str,
        item_id: str,
        context: RequestContext,
    ) -> None:
        await self._load_thread_row(thread_id, context)
        async with self._connect() as db:
            await db.execute(
                "DELETE FROM thread_items WHERE id = ? AND thread_id = ?",
                (item_id, thread_id),
            )
            await db.commit()

    async def create_attachment(
        self,
        input: AttachmentCreateParams,
        context: RequestContext,
    ) -> Attachment:
        attachment = FileAttachment(
            id=self.generate_attachment_id(input.mime_type, context),
            name=input.name,
            mime_type=input.mime_type,
            metadata={"size": input.size},
        )
        await self.save_attachment(attachment, context)
        return attachment

    async def load_all_thread_items(
        self,
        thread_id: str,
        context: RequestContext,
    ) -> list[ThreadItem]:
        await self._load_thread_row(thread_id, context)
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT payload_json
                FROM thread_items
                WHERE thread_id = ?
                ORDER BY sort_index ASC
                """,
                (thread_id,),
            )
            rows = await cursor.fetchall()
        return [THREAD_ITEM_ADAPTER.validate_json(row["payload_json"]) for row in rows]

    async def record_action_receipt(
        self,
        *,
        thread_id: str,
        receipt_key: str,
        action_type: str,
        context: RequestContext,
    ) -> bool:
        await self._load_thread_row(thread_id, context)
        try:
            async with self._connect() as db:
                await db.execute(
                    """
                    INSERT INTO action_receipts (receipt_key, thread_id, owner_user_id, action_type)
                    VALUES (?, ?, ?, ?)
                    """,
                    (receipt_key, thread_id, context.user_id, action_type),
                )
                await db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def _load_thread_row(
        self,
        thread_id: str,
        context: RequestContext,
    ) -> aiosqlite.Row:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT owner_user_id, payload_json FROM threads WHERE id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            raise NotFoundError(f"Thread {thread_id} was not found")
        self._assert_owner(row["owner_user_id"], context)
        return row

    async def _load_thread_owner(self, thread_id: str) -> str | None:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT owner_user_id FROM threads WHERE id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
        return row["owner_user_id"] if row else None

    async def _attachment_owner(self, attachment_id: str) -> str | None:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT owner_user_id FROM attachments WHERE id = ?",
                (attachment_id,),
            )
            row = await cursor.fetchone()
        return row["owner_user_id"] if row else None

    async def _lookup_item_sort_index(self, thread_id: str, item_id: str) -> int:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT sort_index FROM thread_items WHERE thread_id = ? AND id = ?",
                (thread_id, item_id),
            )
            row = await cursor.fetchone()
        if row is None:
            raise NotFoundError(f"Item {item_id} was not found")
        return int(row["sort_index"])

    async def _lookup_thread_sort_index(self, thread_id: str) -> int:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT rowid FROM threads WHERE id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            raise NotFoundError(f"Thread {thread_id} was not found")
        return int(row["rowid"])

    def _assert_owner(self, owner_user_id: str, context: RequestContext) -> None:
        if context.role == "agent":
            return
        if owner_user_id != context.user_id:
            raise NotFoundError("Thread was not found")

    @asynccontextmanager
    async def _connect(self):
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        try:
            yield db
        finally:
            await db.close()
