from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import AssistantSettings, LengthProfile, ModelSettings, RuntimeSettings, SummaryLength

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "assistant.yaml"


def load_settings(config_path: str | Path | None = None) -> AssistantSettings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    runtime_raw = raw_config.get("runtime", {})
    length_profiles_raw = raw_config.get("length_profiles", {})
    models_raw = raw_config.get("models", {})

    runtime = RuntimeSettings(
        device=str(runtime_raw.get("device", "auto")),
        qa_confidence_threshold=float(runtime_raw.get("qa_confidence_threshold", 0.2)),
    )

    length_profiles = {
        SummaryLength.from_value(name): LengthProfile(
            label=SummaryLength.from_value(name),
            min_words=int(payload["min_words"]),
            max_words=int(payload["max_words"]),
            instructions=str(payload["instructions"]),
        )
        for name, payload in length_profiles_raw.items()
    }

    if set(length_profiles) != set(SummaryLength):
        missing = sorted(item.value for item in set(SummaryLength) - set(length_profiles))
        raise ValueError(f"Missing length profile definitions: {', '.join(missing)}")

    summarizer = _build_model_settings(models_raw, "summarizer")
    refiner = _build_model_settings(models_raw, "refiner")
    qa = _build_model_settings(models_raw, "qa")

    return AssistantSettings(
        runtime=runtime,
        length_profiles=length_profiles,
        summarizer=summarizer,
        refiner=refiner,
        qa=qa,
    )


def _build_model_settings(models_raw: dict[str, dict], key: str) -> ModelSettings:
    if key not in models_raw:
        raise ValueError(f"Missing model configuration for '{key}'")

    payload = models_raw[key]
    return ModelSettings(
        model_name=str(payload["model_name"]),
        task=str(payload["task"]),
        generation_kwargs=dict(payload.get("generation_kwargs", {})),
        max_input_tokens=int(payload.get("max_input_tokens", 768)),
        chunk_overlap_tokens=int(payload.get("chunk_overlap_tokens", 64)),
    )
