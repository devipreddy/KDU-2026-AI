from __future__ import annotations

from pathlib import Path

from .classification import QueryClassifier
from .config import ConfigManager
from .llm_provider import MockLLMProvider, OpenAICompatibleProvider
from .models import FixItConfig, SupportRequest, SupportResponse
from .prompts import PromptManager
from .routing import QueryRouter
from .storage import SQLiteStateStore


class FixItService:
    def __init__(self, config_path: str | Path, provider_override=None) -> None:
        self.project_root = Path(config_path).resolve().parent.parent
        self.config_manager = ConfigManager(config_path)
        self.provider_override = provider_override
        self._config: FixItConfig | None = None
        self._router: QueryRouter | None = None
        self.reload(force=True)

    @property
    def config(self) -> FixItConfig:
        if self._config is None:
            raise RuntimeError("Service is not initialized.")
        return self._config

    def reload(self, force: bool = False) -> None:
        config = self.config_manager.load(force=force)
        if not force and self._config is config and self._router is not None:
            return
        store = SQLiteStateStore(self.project_root / config.app.state_db_path)
        prompt_manager = PromptManager(config.prompts, self.project_root)
        classifier = QueryClassifier(config.classification)

        if self.provider_override is not None:
            provider = self.provider_override
        else:
            provider = MockLLMProvider() if config.llm.dry_run else OpenAICompatibleProvider(config.llm)

        self._config = config
        self._router = QueryRouter(
            config=config,
            classifier=classifier,
            prompt_manager=prompt_manager,
            provider=provider,
            store=store,
        )

    def process(self, request: SupportRequest) -> SupportResponse:
        if self.config.feature_flags.enable_hot_reload:
            self.reload(force=False)
        if self._router is None:
            raise RuntimeError("Router is not initialized.")
        return self._router.handle(request)
