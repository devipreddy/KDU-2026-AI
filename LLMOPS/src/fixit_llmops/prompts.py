from __future__ import annotations

import re
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

from .models import PromptAsset, PromptConfig


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _version_key(version: str) -> tuple[int, str]:
    match = re.match(r"v(\d+)", version.lower())
    if match:
        return int(match.group(1)), version
    return 0, version


class PromptManager:
    def __init__(self, prompt_config: PromptConfig, project_root: Path) -> None:
        self.prompt_config = prompt_config
        self.prompt_root = (project_root / prompt_config.root_dir).resolve()
        self._cache: dict[str, list[PromptAsset]] = {}
        self.reload()

    def reload(self) -> None:
        prompts: dict[str, list[PromptAsset]] = {}
        for file_path in sorted(self.prompt_root.glob("*/*.yaml")):
            with file_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            asset = PromptAsset.model_validate(loaded)
            prompts.setdefault(asset.category, []).append(asset)

        for category_assets in prompts.values():
            category_assets.sort(key=lambda item: _version_key(item.version), reverse=True)
        self._cache = prompts

    def get_prompt(self, category: str) -> PromptAsset:
        candidates = self._cache.get(category) or self._cache.get(self.prompt_config.fallback_category, [])
        if not candidates:
            raise FileNotFoundError(f"No prompts found for category '{category}' or fallback category.")

        stable_candidates = [item for item in candidates if item.status == "stable"]
        eligible = stable_candidates or candidates

        if self.prompt_config.selection_strategy == "best_performing":
            return max(eligible, key=lambda item: item.selection_score)
        return sorted(eligible, key=lambda item: _version_key(item.version), reverse=True)[0]

    def render(self, prompt: PromptAsset, context: dict[str, Any]) -> str:
        formatter = Formatter()
        used_context = SafeFormatDict(context)
        return formatter.vformat(prompt.template, (), used_context)

