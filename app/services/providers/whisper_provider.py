from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from app.core.config import Settings


class WhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline = None
        self._lock = threading.Lock()

    def transcribe(self, file_path: Path) -> dict[str, Any]:
        recognizer = self._get_pipeline()
        audio_array, sampling_rate = librosa.load(str(file_path), sr=16000, mono=True)
        try:
            result = recognizer({"array": audio_array, "sampling_rate": sampling_rate}, return_timestamps=True)
        except TypeError:
            result = recognizer({"array": audio_array, "sampling_rate": sampling_rate})
        return {
            "text": str(result.get("text", "")).strip(),
            "language": str(result.get("language", "unknown")),
            "sampling_rate": sampling_rate,
            "chunks": self._normalize_chunks(result.get("chunks", [])),
        }

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            torch_dtype = getattr(torch, self.settings.whisper_torch_dtype, torch.float32)
            device = 0 if self.settings.whisper_device.startswith("cuda") and torch.cuda.is_available() else -1
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.settings.whisper_model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            processor = AutoProcessor.from_pretrained(self.settings.whisper_model_id)
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=self.settings.whisper_chunk_length_s,
                batch_size=self.settings.whisper_batch_size,
                torch_dtype=torch_dtype,
                device=device,
            )
        return self._pipeline

    def _normalize_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for chunk in chunks or []:
            timestamp = chunk.get("timestamp") or (None, None)
            start, end = timestamp if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2 else (None, None)
            normalized.append(
                {
                    "text": str(chunk.get("text", "")).strip(),
                    "start": float(start) if start is not None else None,
                    "end": float(end) if end is not None else None,
                }
            )
        return normalized
