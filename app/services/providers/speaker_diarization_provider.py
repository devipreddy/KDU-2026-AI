from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from app.core.config import Settings


class SpeakerDiarizer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._classifier = None
        self._lock = threading.Lock()

    def annotate(self, file_path: Path, transcript: dict[str, Any]) -> dict[str, Any]:
        chunks = self._normalize_chunks(transcript.get("chunks") or [], transcript.get("text", ""))
        if not chunks:
            return self._single_speaker_output([], transcript.get("text", ""), method="single_speaker_text")

        if not self.settings.speaker_diarization_enabled:
            return self._single_speaker_output(chunks, transcript.get("text", ""), method="single_speaker_disabled")

        try:
            labels = self._cluster_segments(file_path, chunks)
            segments = self._merge_segments(chunks, labels)
            return self._format_output(segments, method="speechbrain_ecapa")
        except Exception as exc:  # pragma: no cover - external model/runtime branch
            return self._single_speaker_output(
                chunks,
                transcript.get("text", ""),
                method="single_speaker_fallback",
                error=str(exc),
            )

    def _cluster_segments(self, file_path: Path, chunks: list[dict[str, Any]]) -> list[int]:
        classifier = self._get_classifier()
        audio_array, sampling_rate = librosa.load(str(file_path), sr=16000, mono=True)
        min_samples = int(sampling_rate * self.settings.speaker_diarization_min_segment_seconds)

        embeddings: list[np.ndarray] = []
        segment_indexes: list[int] = []
        for index, chunk in enumerate(chunks):
            start = chunk["start"]
            end = chunk["end"]
            if start is None or end is None or end <= start:
                continue
            start_index = int(start * sampling_rate)
            end_index = int(end * sampling_rate)
            waveform = audio_array[start_index:end_index]
            if waveform.size < min_samples:
                continue
            embedding = self._encode_segment(classifier, waveform)
            embeddings.append(embedding)
            segment_indexes.append(index)

        if len(embeddings) <= 1:
            return [0] * len(chunks)

        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self.settings.speaker_diarization_distance_threshold,
        )
        assigned = clustering.fit_predict(np.vstack(embeddings))
        labels: list[int | None] = [None] * len(chunks)
        for chunk_index, label in zip(segment_indexes, assigned, strict=False):
            labels[chunk_index] = int(label)

        last_label = 0
        for index, label in enumerate(labels):
            if label is not None:
                last_label = label
                continue
            labels[index] = last_label

        next_label = 0
        for index in range(len(labels) - 1, -1, -1):
            if labels[index] is not None:
                next_label = int(labels[index])
                continue
            labels[index] = next_label

        return [int(label or 0) for label in labels]

    def _encode_segment(self, classifier, waveform: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        embedding = classifier.encode_batch(tensor)
        return embedding.squeeze().detach().cpu().numpy().reshape(-1)

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier

        with self._lock:
            if self._classifier is not None:
                return self._classifier

            from speechbrain.inference.speaker import EncoderClassifier

            device = "cuda" if self.settings.whisper_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
            kwargs = {
                "source": self.settings.speaker_diarization_model_id,
                "savedir": str(self.settings.data_dir / "speaker_model"),
                "run_opts": {"device": device},
            }
            if self.settings.huggingface_token:
                kwargs["use_auth_token"] = self.settings.huggingface_token
            self._classifier = EncoderClassifier.from_hparams(**kwargs)
        return self._classifier

    def _normalize_chunks(self, chunks: list[dict[str, Any]], full_text: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            start = chunk.get("start")
            end = chunk.get("end")
            if not text:
                continue
            normalized.append(
                {
                    "text": text,
                    "start": float(start) if start is not None else None,
                    "end": float(end) if end is not None else None,
                }
            )

        if normalized:
            return normalized

        if full_text.strip():
            return [{"text": full_text.strip(), "start": None, "end": None}]
        return []

    def _merge_segments(self, chunks: list[dict[str, Any]], labels: list[int]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for chunk, label in zip(chunks, labels, strict=False):
            speaker = f"Speaker {label + 1}"
            if merged and merged[-1]["speaker"] == speaker:
                merged[-1]["text"] = f"{merged[-1]['text']} {chunk['text']}".strip()
                if chunk["end"] is not None:
                    merged[-1]["end"] = chunk["end"]
                continue
            merged.append(
                {
                    "speaker": speaker,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                }
            )
        return merged

    def _format_output(self, segments: list[dict[str, Any]], method: str, error: str | None = None) -> dict[str, Any]:
        formatted_lines = []
        speaker_names = []
        for segment in segments:
            speaker_names.append(segment["speaker"])
            time_label = self._format_time_range(segment.get("start"), segment.get("end"))
            prefix = f"{segment['speaker']} {time_label}:".strip() if time_label else f"{segment['speaker']}:"
            formatted_lines.append(f"{prefix} {segment['text']}".strip())

        unique_speakers = []
        for speaker in speaker_names:
            if speaker not in unique_speakers:
                unique_speakers.append(speaker)

        payload = {
            "formatted_transcript": "\n\n".join(formatted_lines).strip(),
            "speaker_segments": segments,
            "speaker_count": len(unique_speakers) or 1,
            "diarization_method": method,
        }
        if error:
            payload["diarization_error"] = error
        return payload

    def _single_speaker_output(
        self,
        chunks: list[dict[str, Any]],
        full_text: str,
        method: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        if not chunks and full_text.strip():
            chunks = [{"text": full_text.strip(), "start": None, "end": None}]
        segments = [
            {
                "speaker": "Speaker 1",
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "text": chunk.get("text", ""),
            }
            for chunk in chunks
        ]
        return self._format_output(segments, method=method, error=error)

    def _format_time_range(self, start: float | None, end: float | None) -> str:
        if start is None and end is None:
            return ""
        start_label = self._format_seconds(start or 0.0)
        end_label = self._format_seconds(end or start or 0.0)
        return f"[{start_label} - {end_label}]"

    def _format_seconds(self, value: float) -> str:
        total_seconds = max(0, int(round(value)))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
