from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

from app.core.logging import session_id_var, trace_id_var
from app.core.models import ConversationMessage, LoggedEvent, SessionPhase, SessionSnapshot
from app.services.audio import AudioTurnBuffer, decode_audio_chunk, encode_audio_segment, pcm_to_wav_bytes, split_for_speech
from app.services.monitoring import Metrics
from app.services.openai_provider import OpenAIProvider
from app.services.orchestrator import Coordinator
from app.services.replay import SessionReplayStore

logger = logging.getLogger(__name__)


class VoiceSession:
    def __init__(
        self,
        *,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        trace_id: str,
        coordinator: Coordinator,
        provider: OpenAIProvider,
        replay_store: SessionReplayStore,
        metrics: Metrics,
        audio_buffer: AudioTurnBuffer,
    ) -> None:
        self.websocket = websocket
        self.snapshot = SessionSnapshot(
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
            phase=SessionPhase.LISTENING,
        )
        self.coordinator = coordinator
        self.provider = provider
        self.replay_store = replay_store
        self.metrics = metrics
        self.audio_buffer = audio_buffer
        self.send_lock = asyncio.Lock()
        self.process_lock = asyncio.Lock()
        self.playback_task: asyncio.Task | None = None
        self.processing_task: asyncio.Task | None = None
        self.closed = False
        self.generation_id = 0
        self.current_segments: list[str] = []
        self.spoken_segments: list[str] = []
        self.spoken_segment_indices: set[int] = set()
        self.current_reply_text: str = ""
        self.current_reply_agent: str = "billing"
        self.final_segment_count = 0
        self.generation_done_announced = False

    async def open(self) -> None:
        await self.emit(
            "session.started",
            {
                "session_id": self.snapshot.session_id,
                "trace_id": self.snapshot.trace_id,
                "user_id": self.snapshot.user_id,
                "phase": self.snapshot.phase,
            },
        )

    async def close(self) -> None:
        self.closed = True
        self.snapshot.phase = SessionPhase.CLOSED
        for task in (self.playback_task, self.processing_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "audio.append":
            await self._handle_audio_append(message)
            return
        if message_type == "interrupt":
            await self.request_interrupt("client_interrupt")
            return
        if message_type == "playback.segment_completed":
            await self._mark_segment_completed(message)
            return
        if message_type == "playback.generation_completed":
            await self._finalize_generation(message)
            return
        if message_type == "snapshot.request":
            await self.emit("snapshot", self.snapshot.model_dump(mode="json"))
            return
        if message_type == "ping":
            await self.emit("pong", {"ok": True})

    async def _handle_audio_append(self, message: dict[str, Any]) -> None:
        if self.closed:
            return
        chunk = decode_audio_chunk(message["audio"])
        if self.snapshot.phase == SessionPhase.SPEAKING:
            result = self.audio_buffer.process_speaking_chunk(chunk)
            if result.interrupt_requested:
                await self.request_interrupt("barge_in", seed=result.interrupt_seed or b"")
            return

        result = self.audio_buffer.process_listening_chunk(chunk)
        if result.speech_started:
            await self.emit("state", {"phase": SessionPhase.LISTENING, "speaking": False})
        if result.turn_committed:
            if self.processing_task and not self.processing_task.done():
                return
            self.processing_task = asyncio.create_task(self._process_turn(result.turn_committed))

    async def request_interrupt(self, reason: str, seed: bytes = b"") -> None:
        if self.snapshot.phase != SessionPhase.SPEAKING:
            return
        self.snapshot.metrics.interruptions += 1
        self.metrics.interruptions.inc()
        self.snapshot.phase = SessionPhase.INTERRUPTED
        await self.emit("state", {"phase": self.snapshot.phase, "reason": reason})
        self.generation_id += 1
        if self.playback_task and not self.playback_task.done():
            self.playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.playback_task
        await self.emit("audio.stop", {"reason": reason})
        partial_text = " ".join(
            self.current_segments[index] for index in sorted(self.spoken_segment_indices)
        ).strip()
        if partial_text:
            self.snapshot.recent_messages.append(
                ConversationMessage(
                    role="assistant",
                    text=partial_text,
                    agent=self.current_reply_agent,
                    interrupted=True,
                    delivered=True,
                )
            )
            self.snapshot.conversation_summary, self.snapshot.recent_messages = (
                self.coordinator.pruner.prune(
                    self.snapshot.conversation_summary,
                    self.snapshot.recent_messages,
                )
            )
        self.current_segments = []
        self.spoken_segments = []
        self.spoken_segment_indices = set()
        self.current_reply_text = ""
        self.final_segment_count = 0
        self.generation_done_announced = False
        self.audio_buffer.seed_interrupt_turn(seed)
        self.snapshot.phase = SessionPhase.LISTENING
        await self.emit("state", {"phase": self.snapshot.phase, "resume": True})

    async def _process_turn(self, turn_pcm: bytes) -> None:
        async with self.process_lock:
            trace_id_var.set(self.snapshot.trace_id)
            session_id_var.set(self.snapshot.session_id)
            self.snapshot.phase = SessionPhase.PROCESSING
            await self.emit("state", {"phase": self.snapshot.phase})
            wav_bytes = pcm_to_wav_bytes(turn_pcm, self.audio_buffer.sample_rate_hz)
            transcript, usage = await self.provider.transcribe_audio(wav_bytes)
            self.snapshot.metrics.transcription_calls += 1
            self.metrics.transcription_calls.inc()
            await self.emit(
                "transcript.final",
                {
                    "text": transcript,
                    "usage": usage.model_dump(mode="json") if usage else None,
                },
            )
            if not transcript:
                self.snapshot.phase = SessionPhase.LISTENING
                await self.emit("state", {"phase": self.snapshot.phase})
                return

            outcome = await self.coordinator.handle_turn(self.snapshot, transcript)
            handoff = outcome["handoff"]
            await self.emit("handoff", handoff.model_dump(mode="json"))
            for observation in outcome["observations"]:
                self.metrics.observe_agent(
                    observation.agent,
                    observation.latency_ms,
                    True,
                )
                await self.emit("agent.event", observation.model_dump(mode="json"))
            for worker_result in outcome["worker_results"]:
                self.metrics.observe_agent(
                    worker_result.agent,
                    worker_result.latency_ms,
                    worker_result.success,
                )
                await self.emit("worker.result", worker_result.model_dump(mode="json"))
            if outcome["consensus"] is not None:
                await self.emit("consensus", outcome["consensus"].model_dump(mode="json"))

            self.current_reply_text = outcome["assistant_text"]
            self.current_reply_agent = self.snapshot.current_agent
            self.current_segments = split_for_speech(self.current_reply_text)
            self.spoken_segments = []
            self.spoken_segment_indices = set()
            self.final_segment_count = len(self.current_segments)
            self.generation_done_announced = False
            self.generation_id += 1
            current_generation = self.generation_id
            self.snapshot.phase = SessionPhase.SPEAKING
            await self.emit(
                "assistant.reply",
                {
                    "text": self.current_reply_text,
                    "generation_id": current_generation,
                    "segment_count": self.final_segment_count,
                },
            )
            await self.emit("state", {"phase": self.snapshot.phase})
            self.playback_task = asyncio.create_task(self._stream_segments(current_generation))

    async def _stream_segments(self, generation_id: int) -> None:
        try:
            for index, segment in enumerate(self.current_segments):
                if generation_id != self.generation_id:
                    return
                audio_bytes = await self.provider.synthesize_speech(segment)
                self.snapshot.metrics.tts_calls += 1
                self.metrics.tts_calls.inc()
                await self.emit(
                    "audio.segment",
                    {
                        "generation_id": generation_id,
                        "segment_index": index,
                        "segment_count": self.final_segment_count,
                        "text": segment,
                        "mime_type": "audio/mpeg",
                        "audio": encode_audio_segment(audio_bytes),
                    },
                )
            if generation_id == self.generation_id:
                self.generation_done_announced = True
                await self.emit(
                    "audio.done",
                    {
                        "generation_id": generation_id,
                        "segment_count": self.final_segment_count,
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("tts_stream_failed")
            await self.emit("error", {"message": str(exc), "stage": "tts"})
            self.snapshot.phase = SessionPhase.LISTENING
            await self.emit("state", {"phase": self.snapshot.phase})

    async def _mark_segment_completed(self, message: dict[str, Any]) -> None:
        if message.get("generation_id") != self.generation_id:
            return
        index = int(message.get("segment_index", -1))
        if 0 <= index < len(self.current_segments):
            self.spoken_segment_indices.add(index)
            self.spoken_segments = [
                self.current_segments[segment_index]
                for segment_index in sorted(self.spoken_segment_indices)
            ]

    async def _finalize_generation(self, message: dict[str, Any]) -> None:
        if message.get("generation_id") != self.generation_id:
            return
        if self.current_reply_text:
            self.snapshot.recent_messages.append(
                ConversationMessage(
                    role="assistant",
                    text=self.current_reply_text,
                    agent=self.current_reply_agent,
                    delivered=True,
                )
            )
            self.snapshot.conversation_summary, self.snapshot.recent_messages = (
                self.coordinator.pruner.prune(
                    self.snapshot.conversation_summary,
                    self.snapshot.recent_messages,
                )
            )
        self.current_segments = []
        self.spoken_segments = []
        self.spoken_segment_indices = set()
        self.current_reply_text = ""
        self.final_segment_count = 0
        self.snapshot.phase = SessionPhase.LISTENING
        await self.emit("state", {"phase": self.snapshot.phase})

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, **payload}
        async with self.send_lock:
            await self.websocket.send_json(event)
        self.replay_store.append(
            LoggedEvent(
                trace_id=self.snapshot.trace_id,
                session_id=self.snapshot.session_id,
                event_type=event_type,
                payload=payload,
            )
        )


class VoiceSessionManager:
    def __init__(self, *, coordinator: Coordinator, provider: OpenAIProvider, replay_store: SessionReplayStore, metrics: Metrics, audio_buffer_factory: callable) -> None:
        self.coordinator = coordinator
        self.provider = provider
        self.replay_store = replay_store
        self.metrics = metrics
        self.audio_buffer_factory = audio_buffer_factory
        self.sessions: dict[str, VoiceSession] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> VoiceSession:
        await websocket.accept()
        session_id = f"sess_{uuid4().hex}"
        trace_id = f"trace_{uuid4().hex}"
        session = VoiceSession(
            websocket=websocket,
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
            coordinator=self.coordinator,
            provider=self.provider,
            replay_store=self.replay_store,
            metrics=self.metrics,
            audio_buffer=self.audio_buffer_factory(),
        )
        self.sessions[session_id] = session
        self.metrics.session_count.set(len(self.sessions))
        await session.open()
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        self.sessions.pop(session.snapshot.session_id, None)
        self.metrics.session_count.set(len(self.sessions))
        await session.close()

    def get_snapshot(self, session_id: str) -> dict[str, Any] | None:
        session = self.sessions.get(session_id)
        if not session:
            return None
        return session.snapshot.model_dump(mode="json")
