from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..schemas import OrderIntent, UsageRecord
from ..tracing import traceable

try:
    from langchain_openrouter import ChatOpenRouter
except ImportError:  # pragma: no cover - fallback for environments without langchain-openrouter.
    ChatOpenRouter = None  # type: ignore[assignment]


MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
}

QUOTE_TERMS = ("quote", "price", "market price", "how much", "latest")
BUY_TERMS = ("buy", "purchase", "accumulate")
SELL_TERMS = ("sell", "offload", "exit")
PORTFOLIO_TERMS = ("portfolio", "holdings", "net worth", "summary", "positions")
SYMBOL_PATTERN = re.compile(r"\b[A-Z][A-Z0-9.]{0,12}\b")
NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\b")
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class IntentParsingError(RuntimeError):
    """Raised when the configured LLM parser cannot complete successfully."""


@dataclass
class IntentParser:
    settings: Settings
    supported_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._llm = None
        self._structured_llm = None
        if self.settings.openrouter_enabled and ChatOpenRouter is not None:
            self._llm = ChatOpenRouter(
                openrouter_api_key=self.settings.openrouter_api_key,
                model=self.settings.openrouter_model,
                temperature=0,
                timeout=int(self.settings.openrouter_timeout_seconds * 1000),
                max_retries=self.settings.openrouter_max_retries,
                app_url=self.settings.openrouter_app_url,
                app_title=self.settings.openrouter_app_title,
            )
            try:
                self._structured_llm = self._llm.with_structured_output(
                    OrderIntent,
                    method="function_calling",
                    include_raw=True,
                    strict=True,
                )
            except Exception:
                self._structured_llm = None

    @traceable(name="intent_parse")
    def parse(self, text: str, base_currency: str) -> tuple[OrderIntent, UsageRecord]:
        normalized_text = " ".join(text.strip().split())
        if self._llm is not None:
            try:
                return self._parse_with_llm(normalized_text, base_currency)
            except Exception as exc:
                if self.settings.intent_parser_allow_fallback:
                    intent, usage = self._parse_rule_based(normalized_text)
                    usage.metadata.update(
                        {
                            "fallback": True,
                            "requested_parser": "openrouter",
                            "llm_error_type": type(exc).__name__,
                            "llm_error_message": str(exc),
                        }
                    )
                    return intent, usage
                raise IntentParsingError(self._format_llm_error(exc)) from exc
        return self._parse_rule_based(normalized_text)

    def _parse_rule_based(self, text: str) -> tuple[OrderIntent, UsageRecord]:
        lowered = text.lower()
        symbol = self._extract_symbol(text)
        quantity = self._extract_quantity(text)
        notes: list[str] = []

        if any(term in lowered for term in BUY_TERMS):
            action = "BUY"
            if quantity is None:
                notes.append("Quantity missing from BUY request.")
        elif any(term in lowered for term in SELL_TERMS):
            action = "SELL"
            if quantity is None:
                notes.append("Quantity missing from SELL request.")
        elif any(term in lowered for term in PORTFOLIO_TERMS):
            action = "PORTFOLIO"
        elif any(term in lowered for term in QUOTE_TERMS) or symbol:
            action = "QUOTE"
        else:
            action = "UNKNOWN"
            notes.append("Could not classify the request deterministically.")

        intent = OrderIntent(
            action=action,
            symbol=symbol,
            quantity=quantity,
            confidence=0.95 if action != "UNKNOWN" else 0.0,
            notes=notes,
        )
        usage = UsageRecord(
            source="intent-parser",
            model="rule-based",
            metadata={"fallback": False, "requested_parser": "rule-based"},
        )
        return intent, usage

    def _parse_with_llm(self, text: str, base_currency: str) -> tuple[OrderIntent, UsageRecord]:
        prompt = (
            "You are an intent parser for a deterministic financial workflow.\n"
            "Return JSON only with keys: action, symbol, quantity, confidence, notes.\n"
            "Allowed actions: BUY, SELL, QUOTE, PORTFOLIO, HELP, UNKNOWN.\n"
            "confidence must be a numeric value between 0 and 1.\n"
            "notes must be an array of strings.\n"
            "Do not invent symbols or quantities.\n"
            f"User base currency: {base_currency}\n"
            f"User text: {text}"
        )
        if self._structured_llm is not None:
            structured_result = self._structured_llm.invoke(prompt)
            parsed_intent = structured_result.get("parsed")
            raw_response = structured_result.get("raw")
            parsing_error = structured_result.get("parsing_error")
            if parsed_intent is not None:
                usage = self._build_usage_record(raw_response)
                return parsed_intent, usage
            if raw_response is not None:
                intent = self._parse_llm_response(raw_response)
                usage = self._build_usage_record(raw_response)
                usage.metadata["structured_output_fallback"] = True
                if parsing_error is not None:
                    usage.metadata["structured_output_error"] = str(parsing_error)
                return intent, usage
            if parsing_error is not None:
                raise parsing_error

        response = self._llm.invoke(prompt)
        intent = self._parse_llm_response(response)
        usage = self._build_usage_record(response)
        return intent, usage

    def _parse_llm_response(self, response: Any) -> OrderIntent:
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        payload = self._extract_json(content)
        normalized_payload = self._normalize_intent_payload(payload)
        return OrderIntent.model_validate(normalized_payload)

    def _normalize_intent_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "UNKNOWN")).strip().upper() or "UNKNOWN"
        if action not in {"BUY", "SELL", "QUOTE", "PORTFOLIO", "HELP", "UNKNOWN"}:
            action = "UNKNOWN"

        symbol = payload.get("symbol")
        if symbol is not None:
            symbol = str(symbol).strip().upper() or None

        quantity = self._coerce_float(payload.get("quantity"))
        confidence = self._coerce_confidence(payload.get("confidence"))
        notes = self._coerce_notes(payload.get("notes"))

        return {
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "confidence": confidence,
            "notes": notes,
        }

    def _coerce_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            value = cleaned
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_confidence(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if not cleaned:
                return 0.0
            confidence_aliases = {
                "very low": 0.1,
                "low": 0.3,
                "medium": 0.6,
                "med": 0.6,
                "moderate": 0.6,
                "high": 0.9,
                "very high": 0.98,
            }
            if cleaned in confidence_aliases:
                return confidence_aliases[cleaned]
            try:
                return max(0.0, min(1.0, float(cleaned)))
            except ValueError:
                return 0.0
        return 0.0

    def _coerce_notes(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        cleaned = str(value).strip()
        return [cleaned] if cleaned else []

    def _extract_symbol(self, text: str) -> str | None:
        upper_text = text.upper()
        for symbol in self.supported_symbols:
            if symbol in upper_text:
                return symbol

        candidates = SYMBOL_PATTERN.findall(upper_text)
        ignored = {"BUY", "SELL", "PRICE", "QUOTE", "PORTFOLIO", "INR", "USD", "EUR", "HELP"}
        for candidate in candidates:
            if candidate not in ignored:
                return candidate
        return None

    def _extract_quantity(self, text: str) -> float | None:
        match = NUMBER_PATTERN.search(text)
        return float(match.group(1)) if match else None

    def _extract_json(self, content: str) -> dict[str, Any]:
        match = JSON_BLOCK_PATTERN.search(content.strip())
        if not match:
            raise ValueError("Model response did not contain a JSON object.")
        return json.loads(match.group(0))

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING_PER_1M.get(self.settings.openrouter_model)
        if pricing is None:
            return 0.0
        input_rate, output_rate = pricing
        cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
        return round(cost, 8)

    def _build_usage_record(self, response: Any) -> UsageRecord:
        response_metadata = getattr(response, "response_metadata", {}) or {}
        usage_metadata = getattr(response, "usage_metadata", {}) or response_metadata.get("token_usage", {})
        if not usage_metadata and isinstance(response_metadata.get("usage"), dict):
            usage_metadata = response_metadata["usage"]
        input_tokens = int(usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0)) or 0)
        output_tokens = int(usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0)) or 0)
        total_tokens = int(usage_metadata.get("total_tokens", input_tokens + output_tokens) or 0)
        estimated_cost_usd = self._extract_cost(response) or self._estimate_cost(input_tokens, output_tokens)
        return UsageRecord(
            source="intent-parser",
            model=self.settings.openrouter_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            metadata={"fallback": False, "requested_parser": "openrouter"},
        )

    def _extract_cost(self, response: Any) -> float | None:
        response_metadata = getattr(response, "response_metadata", {}) or {}
        for key in ("usage", "token_usage", "usage_metadata"):
            usage_payload = response_metadata.get(key, {})
            if isinstance(usage_payload, dict):
                cost = usage_payload.get("cost")
                if cost is not None:
                    try:
                        return round(float(cost), 8)
                    except (TypeError, ValueError):
                        return None
        return None

    def _format_llm_error(self, exc: Exception) -> str:
        message = str(exc)
        if "insufficient_quota" in message or "current quota" in message.lower():
            return (
                "OpenRouter intent parsing failed because the configured API project has insufficient quota. "
                "Update billing/quota or set INTENT_PARSER_ALLOW_FALLBACK=true if you want rule-based fallback."
            )
        if any(term in message.lower() for term in ("timed out", "timeout", "connecttimeout", "readtimeout")):
            return (
                "OpenRouter intent parsing failed because the provider request timed out. "
                "This is usually transient; retry the same command or increase OPENROUTER_TIMEOUT_SECONDS / OPENROUTER_MAX_RETRIES. "
                "Set INTENT_PARSER_ALLOW_FALLBACK=true if you want rule-based fallback."
            )
        return (
            f"OpenRouter intent parsing failed: {type(exc).__name__}: {message}. "
            "Set INTENT_PARSER_ALLOW_FALLBACK=true if you want rule-based fallback."
        )
