from __future__ import annotations

from pathlib import Path

from app.services.providers.speaker_diarization_provider import SpeakerDiarizer
from app.services.providers.whisper_provider import WhisperTranscriber
from app.services.types import ExtractedSection, ExtractionResult


class AudioProcessor:
    def __init__(self, transcriber: WhisperTranscriber, diarizer: SpeakerDiarizer) -> None:
        self.transcriber = transcriber
        self.diarizer = diarizer

    def extract(self, file_path: Path) -> ExtractionResult:
        transcript = self.transcriber.transcribe(file_path)
        diarization = self.diarizer.annotate(file_path, transcript)
        section = ExtractedSection(
            text=diarization["formatted_transcript"] or transcript["text"],
            source="whisper_local",
            metadata={
                "language": transcript.get("language", "unknown"),
                "speaker_count": diarization.get("speaker_count", 1),
                "diarization_method": diarization.get("diarization_method"),
            },
        )
        return ExtractionResult(
            sections=[section],
            metadata={
                "language": transcript.get("language", "unknown"),
                "sampling_rate": transcript.get("sampling_rate"),
                "speaker_count": diarization.get("speaker_count", 1),
                "speaker_segments": diarization.get("speaker_segments", []),
                "diarization_method": diarization.get("diarization_method"),
                "diarization_error": diarization.get("diarization_error"),
            },
            usage_entries=[],
        )
