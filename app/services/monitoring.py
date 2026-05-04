from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.session_count = Gauge(
            "voice_sessions_active",
            "Active voice sessions",
            registry=self.registry,
        )
        self.agent_calls = Counter(
            "voice_agent_calls_total",
            "Agent calls",
            labelnames=("agent", "outcome"),
            registry=self.registry,
        )
        self.agent_latency = Histogram(
            "voice_agent_latency_seconds",
            "Agent call latency",
            labelnames=("agent",),
            buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.transcription_calls = Counter(
            "voice_transcriptions_total",
            "Transcription calls",
            registry=self.registry,
        )
        self.tts_calls = Counter(
            "voice_tts_total",
            "TTS calls",
            registry=self.registry,
        )
        self.interruptions = Counter(
            "voice_interruptions_total",
            "Playback interruptions",
            registry=self.registry,
        )

    def observe_agent(self, agent: str, latency_ms: int, success: bool) -> None:
        self.agent_calls.labels(agent=agent, outcome="success" if success else "failure").inc()
        self.agent_latency.labels(agent=agent).observe(latency_ms / 1000)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
