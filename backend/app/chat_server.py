from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents import Agent, RunContextWrapper, Runner, function_tool
from chatkit.actions import Action
from chatkit.agents import AgentContext, ThreadItemConverter, simple_to_agent_input, stream_agent_response
from chatkit.errors import CustomStreamError
from chatkit.server import ChatKitServer, stream_widget
from chatkit.store import NotFoundError
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    HiddenContextItem,
    NoticeEvent,
    ProgressUpdateEvent,
    ThreadItemDoneEvent,
    ThreadMetadata,
    WidgetItem,
)
from chatkit.widgets import WidgetTemplate

from .actions import (
    BookNowClientAction,
    BookNowServerAction,
    RequestHandoffAction,
    ResumeAIAction,
    parse_app_action,
)
from .catalog import TravelCatalog, TravelOffer
from .config import Settings
from .context import RequestContext
from .realtime import RealtimeHub
from .store import SQLiteChatStore


class TravelChatKitServer(ChatKitServer[RequestContext]):
    def __init__(
        self,
        *,
        settings: Settings,
        store: SQLiteChatStore,
        catalog: TravelCatalog,
        hub: RealtimeHub,
    ):
        super().__init__(store=store, attachment_store=store)
        self.settings = settings
        self.catalog = catalog
        self.hub = hub
        widget_dir = Path(__file__).resolve().parent / "widgets"
        self.booking_options_template = WidgetTemplate.from_file(
            str(widget_dir / "booking_options.widget")
        )
        self.booking_confirmation_template = WidgetTemplate.from_file(
            str(widget_dir / "booking_confirmation.widget")
        )
        self.handoff_status_template = WidgetTemplate.from_file(
            str(widget_dir / "handoff_status.widget")
        )
        self.thread_converter = ThreadItemConverter()
        self.use_mock_model = self.settings.use_mock_model or not bool(
            self.settings.openai_api_key
        )
        self.assistant_agent = self._build_agent()

    def _build_agent(self) -> Agent[AgentContext[RequestContext]]:
        @function_tool(
            description_override=(
                "Search the curated travel catalog and stream interactive booking widgets "
                "for the customer."
            )
        )
        async def search_trip_options(
            ctx: RunContextWrapper[AgentContext[RequestContext]],
            destination_or_request: str,
        ) -> str:
            offers = self.catalog.search(destination_or_request)
            if not offers:
                return "No travel options were found in the curated catalog."

            ctx.context.thread.metadata["last_offer_ids"] = [offer.id for offer in offers]
            ctx.context.thread.metadata["last_offer_query"] = destination_or_request

            widget = self.booking_options_template.build(
                {
                    "query": destination_or_request,
                    "offers": [
                        {
                            "id": offer.id,
                            "title": offer.title,
                            "destination": offer.destination,
                            "origin": offer.origin,
                            "price_display": offer.price_display,
                            "nights": offer.nights,
                            "airline": offer.airline,
                            "hotel": offer.hotel,
                            "highlights": offer.highlights,
                        }
                        for offer in offers
                    ],
                }
            )
            await ctx.context.stream_widget(
                widget,
                copy_text="Interactive travel options were shown to the customer.",
            )
            return " ".join(
                f"{offer.title} for {offer.price_display}" for offer in offers
            )

        instructions = (
            "You are Voyager, a concise travel booking assistant. "
            "When the user asks about destinations, flights, hotels, or booking help, "
            "use the search_trip_options tool so the UI can show live offers. "
            "Keep answers short, practical, and oriented around next actions."
        )
        return Agent(
            name="Voyager",
            model=self.settings.openai_model,
            instructions=instructions,
            tools=[search_trip_options],
        )

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: Any,
        context: RequestContext,
    ) -> AsyncIterator[Any]:
        mode = self._conversation_mode(thread)
        if mode != "ai":
            message = (
                "A human travel specialist is handling this chat now."
                if mode == "human"
                else "A human travel specialist has been requested. AI replies are paused."
            )
            yield NoticeEvent(level="info", message=message)
            return

        if self.use_mock_model:
            async for event in self._respond_with_mock(thread, input_user_message, context):
                yield event
            return

        async for event in self._respond_with_openai(thread, context):
            yield event

    async def action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: RequestContext,
    ) -> AsyncIterator[Any]:
        app_action = parse_app_action(action)

        if app_action.type == "travel.book_now.confirm":
            async for event in self._handle_book_now(thread, app_action, context):
                yield event
            return

        if app_action.type == "support.request_handoff":
            async for event in self._handle_request_handoff(thread, app_action, context):
                yield event
            return

        if app_action.type == "support.resume_ai":
            async for event in self._handle_resume_ai(thread, app_action, context):
                yield event
            return

        raise CustomStreamError(message="Unsupported action received from the client")

    async def claim_handoff(
        self,
        *,
        thread_id: str,
        agent_name: str,
        context: RequestContext,
    ) -> None:
        thread = await self.store.load_thread(thread_id, context)
        thread.metadata["conversation_mode"] = "human"
        thread.metadata["assigned_agent_name"] = agent_name
        thread.metadata["claimed_at"] = datetime.now(UTC).isoformat()
        await self.store.save_thread(thread, context)
        await self.store.add_thread_item(
            thread.id,
            HiddenContextItem(
                id=self.store.generate_item_id("sdk_hidden_context", thread, context),
                thread_id=thread.id,
                created_at=datetime.now(),
                content=f"<HUMAN_HANDOFF>The conversation was claimed by {agent_name}.</HUMAN_HANDOFF>",
            ),
            context,
        )
        await self.hub.broadcast_thread(
            thread.id,
            "handoff.claimed",
            {
                "assigned_agent_name": agent_name,
                "conversation_mode": "human",
            },
        )
        await self.hub.broadcast_queue("handoff.queue.updated", {})

    async def release_handoff(
        self,
        *,
        thread_id: str,
        resume_ai: bool,
        context: RequestContext,
    ) -> None:
        thread = await self.store.load_thread(thread_id, context)
        thread.metadata["conversation_mode"] = "ai" if resume_ai else "handoff_requested"
        thread.metadata["assigned_agent_name"] = None
        await self.store.save_thread(thread, context)
        await self.store.add_thread_item(
            thread.id,
            HiddenContextItem(
                id=self.store.generate_item_id("sdk_hidden_context", thread, context),
                thread_id=thread.id,
                created_at=datetime.now(),
                content=(
                    "<HUMAN_HANDOFF>The human travel specialist released the thread. "
                    f"AI resume is {'enabled' if resume_ai else 'disabled'}.</HUMAN_HANDOFF>"
                ),
            ),
            context,
        )
        await self.hub.broadcast_thread(
            thread.id,
            "handoff.released",
            {"conversation_mode": thread.metadata["conversation_mode"]},
        )
        await self.hub.broadcast_queue("handoff.queue.updated", {})

    async def post_human_message(
        self,
        *,
        thread_id: str,
        text: str,
        context: RequestContext,
    ) -> AssistantMessageItem:
        thread = await self.store.load_thread(thread_id, context)
        if self._conversation_mode(thread) != "human":
            raise CustomStreamError(
                message="Human messages can only be sent while the thread is in human handoff mode.",
                allow_retry=False,
            )

        item = self._assistant_message_item(
            thread=thread,
            text=text,
            context=context,
        )
        await self.store.add_thread_item(thread.id, item, context)
        await self.hub.broadcast_thread(
            thread.id,
            "thread.item.added",
            {"item_type": item.type, "item_id": item.id},
        )
        return item

    async def thread_summary(
        self,
        *,
        thread_id: str,
        context: RequestContext,
    ) -> dict[str, Any]:
        thread = await self.store.load_thread(thread_id, context)
        last_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=1,
            order="desc",
            context=context,
        )
        last_message_preview = None
        if last_page.data:
            last_item = last_page.data[0]
            if hasattr(last_item, "content") and getattr(last_item, "content", None):
                first_content = last_item.content[0]
                last_message_preview = getattr(first_content, "text", None)
            elif isinstance(last_item, WidgetItem):
                last_message_preview = last_item.copy_text or "Interactive widget"

        return {
            "thread_id": thread.id,
            "title": thread.title,
            "conversation_mode": self._conversation_mode(thread),
            "assigned_agent_name": thread.metadata.get("assigned_agent_name"),
            "claimed_at": thread.metadata.get("claimed_at"),
            "last_message_preview": last_message_preview,
            "last_updated_at": last_page.data[0].created_at.isoformat()
            if last_page.data
            else None,
        }

    async def _respond_with_openai(
        self,
        thread: ThreadMetadata,
        context: RequestContext,
    ) -> AsyncIterator[Any]:
        history = await self.store.load_all_thread_items(thread.id, context)
        agent_input = await simple_to_agent_input(history)
        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )
        result = Runner.run_streamed(
            self.assistant_agent,
            agent_input,
            context=agent_context,
            max_turns=6,
        )
        async for event in stream_agent_response(agent_context, result):
            yield event

    async def _respond_with_mock(
        self,
        thread: ThreadMetadata,
        input_user_message: Any,
        context: RequestContext,
    ) -> AsyncIterator[Any]:
        text = self._extract_user_text(input_user_message)
        lowered = text.lower()
        if any(
            keyword in lowered
            for keyword in ("book", "trip", "flight", "hotel", "travel", "vacation")
        ):
            offers = self.catalog.search(text)
            thread.metadata["last_offer_ids"] = [offer.id for offer in offers]
            thread.metadata["last_offer_query"] = text
            yield ProgressUpdateEvent(text="Searching curated travel offers...")
            widget = self.booking_options_template.build(
                {
                    "query": text,
                    "offers": [
                        {
                            "id": offer.id,
                            "title": offer.title,
                            "destination": offer.destination,
                            "origin": offer.origin,
                            "price_display": offer.price_display,
                            "nights": offer.nights,
                            "airline": offer.airline,
                            "hotel": offer.hotel,
                            "highlights": offer.highlights,
                        }
                        for offer in offers
                    ],
                }
            )
            async for event in stream_widget(
                thread,
                widget,
                copy_text="Interactive travel options were shown to the customer.",
                generate_id=lambda item_type: self.store.generate_item_id(
                    item_type, thread, context
                ),
            ):
                yield event

            yield ThreadItemDoneEvent(
                item=self._assistant_message_item(
                    thread=thread,
                    text=(
                        "I found a few curated options and surfaced them in the chat. "
                        "Use Book Now to continue without sending a visible message."
                    ),
                    context=context,
                )
            )
            return

        yield ThreadItemDoneEvent(
            item=self._assistant_message_item(
                thread=thread,
                text=(
                    "I can help with trip planning, destination ideas, and booking flows. "
                    "Try asking for a city or type of getaway."
                ),
                context=context,
            )
        )

    async def _handle_book_now(
        self,
        thread: ThreadMetadata,
        action: BookNowServerAction,
        context: RequestContext,
    ) -> AsyncIterator[Any]:
        visible_offer_ids = thread.metadata.get("last_offer_ids") or []
        if action.payload.offer_id not in visible_offer_ids:
            raise CustomStreamError(
                message="That offer is no longer active for this session.",
                allow_retry=False,
            )

        receipt_key = f"{thread.id}:{action.payload.idempotency_key}"
        receipt_created = await self.store.record_action_receipt(
            thread_id=thread.id,
            receipt_key=receipt_key,
            action_type=action.type,
            context=context,
        )
        if not receipt_created:
            yield NoticeEvent(
                level="info",
                message="That booking action was already processed. No duplicate reservation was created.",
            )
            return

        offer = self.catalog.get(action.payload.offer_id)
        if offer is None:
            raise CustomStreamError(message="The selected offer could not be found.", allow_retry=False)

        thread.metadata["booking_state"] = {
            "offer_id": offer.id,
            "status": "hold_started",
            "held_at": datetime.now(UTC).isoformat(),
        }
        await self.store.add_thread_item(
            thread.id,
            HiddenContextItem(
                id=self.store.generate_item_id("sdk_hidden_context", thread, context),
                thread_id=thread.id,
                created_at=datetime.now(),
                content=(
                    "<USER_ACTION>The user clicked Book Now for offer "
                    f"{offer.id} ({offer.title}).</USER_ACTION>"
                ),
            ),
            context,
        )
        yield ProgressUpdateEvent(text=f"Holding {offer.title}...")
        widget = self.booking_confirmation_template.build(
            {
                "title": offer.title,
                "destination": offer.destination,
                "price_display": offer.price_display,
                "airline": offer.airline,
                "hotel": offer.hotel,
            }
        )
        async for event in stream_widget(
            thread,
            widget,
            copy_text="Booking confirmation details were shown to the customer.",
            generate_id=lambda item_type: self.store.generate_item_id(
                item_type, thread, context
            ),
        ):
            yield event

        yield ThreadItemDoneEvent(
            item=self._assistant_message_item(
                thread=thread,
                text=(
                    f"I started a reservation hold for {offer.title}. "
                    "Because this came from a widget action, it was sent as a hidden event instead of a visible user message."
                ),
                context=context,
            )
        )

    async def _handle_request_handoff(
        self,
        thread: ThreadMetadata,
        action: RequestHandoffAction,
        context: RequestContext,
    ) -> AsyncIterator[Any]:
        mode = self._conversation_mode(thread)
        if mode != "ai":
            yield NoticeEvent(
                level="info",
                message="Human handoff has already been requested or claimed for this thread.",
            )
            return

        thread.metadata["conversation_mode"] = "handoff_requested"
        thread.metadata["handoff_reason"] = action.payload.reason or "Customer request"
        await self.store.add_thread_item(
            thread.id,
            HiddenContextItem(
                id=self.store.generate_item_id("sdk_hidden_context", thread, context),
                thread_id=thread.id,
                created_at=datetime.now(),
                content=(
                    "<HUMAN_HANDOFF>The user requested a human travel specialist. "
                    "Pause AI responses until the handoff is resolved.</HUMAN_HANDOFF>"
                ),
            ),
            context,
        )
        widget = self.handoff_status_template.build(
            {
                "headline": "Human handoff requested",
                "body": "AI replies are paused while a travel specialist joins the conversation.",
                "state": "waiting",
                "agent_name": "",
            }
        )
        async for event in stream_widget(
            thread,
            widget,
            copy_text="Human handoff status was shown to the customer.",
            generate_id=lambda item_type: self.store.generate_item_id(
                item_type, thread, context
            ),
        ):
            yield event
        yield NoticeEvent(
            level="info",
            message="A human travel specialist has been requested. AI responses are paused.",
        )
        await self.hub.broadcast_thread(
            thread.id,
            "handoff.requested",
            {"conversation_mode": "handoff_requested"},
        )
        await self.hub.broadcast_queue("handoff.queue.updated", {})

    async def _handle_resume_ai(
        self,
        thread: ThreadMetadata,
        action: ResumeAIAction,
        context: RequestContext,
    ) -> AsyncIterator[Any]:
        thread.metadata["conversation_mode"] = "ai"
        thread.metadata["assigned_agent_name"] = None
        await self.store.add_thread_item(
            thread.id,
            HiddenContextItem(
                id=self.store.generate_item_id("sdk_hidden_context", thread, context),
                thread_id=thread.id,
                created_at=datetime.now(),
                content=(
                    "<HUMAN_HANDOFF>AI control resumed."
                    f"{' Note: ' + action.payload.note if action.payload.note else ''}</HUMAN_HANDOFF>"
                ),
            ),
            context,
        )
        yield NoticeEvent(level="info", message="AI responses resumed for this thread.")
        await self.hub.broadcast_thread(
            thread.id,
            "handoff.resumed",
            {"conversation_mode": "ai"},
        )
        await self.hub.broadcast_queue("handoff.queue.updated", {})

    def _assistant_message_item(
        self,
        *,
        thread: ThreadMetadata,
        text: str,
        context: RequestContext,
    ) -> AssistantMessageItem:
        return AssistantMessageItem(
            id=self.store.generate_item_id("message", thread, context),
            thread_id=thread.id,
            created_at=datetime.now(),
            content=[AssistantMessageContent(text=text, annotations=[])],
        )

    def _conversation_mode(self, thread: ThreadMetadata) -> str:
        return str(thread.metadata.get("conversation_mode", "ai"))

    def _extract_user_text(self, input_user_message: Any) -> str:
        if input_user_message is None:
            return ""
        content = getattr(input_user_message, "content", []) or []
        text_parts = [getattr(part, "text", "") for part in content if getattr(part, "text", "")]
        return " ".join(text_parts).strip()

