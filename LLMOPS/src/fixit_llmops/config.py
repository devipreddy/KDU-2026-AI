from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import FixItConfig

ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _expand_env_in_string(value: str) -> Any:
    match = ENV_PATTERN.fullmatch(value.strip())
    if match:
        env_name, default = match.groups()
        return os.getenv(env_name, default if default is not None else "")

    def replace_token(token_match: re.Match[str]) -> str:
        env_name, default = token_match.groups()
        return os.getenv(env_name, default if default is not None else "")

    return ENV_PATTERN.sub(replace_token, value)


def _expand_env(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _expand_env(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_expand_env(item) for item in data]
    if isinstance(data, str):
        return _expand_env_in_string(data)
    return data


class ConfigManager:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self._cached: FixItConfig | None = None
        self._last_mtime: float | None = None

    def load(self, force: bool = False) -> FixItConfig:
        current_mtime = self.config_path.stat().st_mtime
        if not force and self._cached is not None and self._last_mtime == current_mtime:
            return self._cached

        with self.config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}

        expanded = _expand_env(raw_config)
        self._cached = FixItConfig.model_validate(expanded)
        self._last_mtime = current_mtime
        return self._cached

    def reload_if_changed(self) -> FixItConfig:
        return self.load(force=False)

