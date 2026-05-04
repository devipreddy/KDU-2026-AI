from __future__ import annotations

import base64
import io
import math
import re
import wave
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class AudioProcessResult:
    speech_started: bool = False
    turn_committed: bytes | None = None
    interrupt_requested: bool = False
    interrupt_seed: bytes | None = None


class AudioTurnBuffer:
    def __init__(
        self,
        *,
        sample_rate_hz: int,
        chunk_ms: int,
        speech_start_threshold: float,
        speech_end_silence_ms: int,
        preroll_ms: int,
        interrupt_threshold: float,
        interrupt_min_chunks: int,
        interrupt_preroll_ms: int,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.chunk_ms = chunk_ms
        self.speech_start_threshold = speech_start_threshold
        self.speech_end_silence_ms = speech_end_silence_ms
        self.preroll_frames = max(1, math.ceil(preroll_ms / chunk_ms))
        self.interrupt_threshold = interrupt_threshold
        self.interrupt_min_chunks = interrupt_min_chunks
        self.interrupt_preroll_frames = max(1, math.ceil(interrupt_preroll_ms / chunk_ms))

        self._speech_active = False
        self._silence_ms = 0
        self._preroll: deque[bytes] = deque(maxlen=self.preroll_frames)
        self._turn_chunks: list[bytes] = []

        self._interrupt_preroll: deque[bytes] = deque(maxlen=self.interrupt_preroll_frames)
        self._interrupt_streak = 0

    def reset_all(self) -> None:
        self._speech_active = False
        self._silence_ms = 0
        self._preroll.clear()
        self._turn_chunks.clear()
        self._interrupt_preroll.clear()
        self._interrupt_streak = 0

    def process_listening_chunk(self, chunk: bytes) -> AudioProcessResult:
        result = AudioProcessResult()
        rms = pcm16le_rms(chunk)
        self._preroll.append(chunk)

        if rms >= self.speech_start_threshold:
            if not self._speech_active:
                self._speech_active = True
                self._silence_ms = 0
                result.speech_started = True
                self._turn_chunks = list(self._preroll)
            else:
                self._silence_ms = 0
            self._turn_chunks.append(chunk)
            return result

        if self._speech_active:
            self._turn_chunks.append(chunk)
            self._silence_ms += self.chunk_ms
            if self._silence_ms >= self.speech_end_silence_ms:
                committed = b"".join(self._turn_chunks)
                self._speech_active = False
                self._silence_ms = 0
                self._turn_chunks = []
                self._preroll.clear()
                result.turn_committed = committed
        return result

    def process_speaking_chunk(self, chunk: bytes) -> AudioProcessResult:
        result = AudioProcessResult()
        self._interrupt_preroll.append(chunk)
        rms = pcm16le_rms(chunk)
        if rms >= self.interrupt_threshold:
            self._interrupt_streak += 1
        else:
            self._interrupt_streak = 0

        if self._interrupt_streak >= self.interrupt_min_chunks:
            result.interrupt_requested = True
            result.interrupt_seed = b"".join(self._interrupt_preroll)
            self._interrupt_streak = 0
            self._interrupt_preroll.clear()
        return result

    def seed_interrupt_turn(self, seed: bytes) -> None:
        self._speech_active = True
        self._silence_ms = 0
        self._turn_chunks = [seed] if seed else []
        self._preroll.clear()
        if seed:
            self._preroll.append(seed)


def pcm16le_rms(chunk: bytes) -> float:
    if len(chunk) < 2:
        return 0.0
    sample_count = len(chunk) // 2
    if sample_count == 0:
        return 0.0
    total = 0.0
    for index in range(0, len(chunk), 2):
        sample = int.from_bytes(chunk[index : index + 2], byteorder="little", signed=True)
        total += sample * sample
    return math.sqrt(total / sample_count)


_sentence_boundary = re.compile(r"(?<=[.!?])\s+")


def split_for_speech(text: str, max_chars: int = 220) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = _sentence_boundary.split(normalized)
    segments: list[str] = []
    current = ""
    for sentence in sentences:
        proposal = sentence if not current else f"{current} {sentence}"
        if len(proposal) <= max_chars:
            current = proposal
            continue
        if current:
            segments.append(current)
        current = sentence
    if current:
        segments.append(current)
    return segments


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate_hz: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def encode_audio_segment(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def decode_audio_chunk(payload: str) -> bytes:
    return base64.b64decode(payload.encode("ascii"))
