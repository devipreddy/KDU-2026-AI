from __future__ import annotations

import hashlib
from typing import Protocol

import httpx

from multimodal_assistant.schemas import WeatherResult


class WeatherClient(Protocol):
    def get_current_weather(self, location: str) -> WeatherResult: ...

    def close(self) -> None: ...


class MockWeatherClient:
    _summaries = [
        "sunny",
        "partly cloudy",
        "cloudy",
        "light rain",
        "clear skies",
        "misty",
        "breezy",
    ]

    def get_current_weather(self, location: str) -> WeatherResult:
        digest = hashlib.sha256(location.lower().encode("utf-8")).digest()
        temperature = round(16 + (digest[0] % 17) + (digest[1] / 255), 1)
        summary = self._summaries[digest[2] % len(self._summaries)]
        return WeatherResult(
            temperature=temperature,
            summary=summary,
            location=location,
        )

    def close(self) -> None:
        return None


class OpenMeteoWeatherClient:
    _weather_codes = {
        0: "clear skies",
        1: "mostly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "light rain",
        63: "rain",
        65: "heavy rain",
        71: "light snow",
        73: "snow",
        75: "heavy snow",
        80: "rain showers",
        81: "showery rain",
        82: "strong rain showers",
        95: "thunderstorm",
    }

    def __init__(self, timeout_seconds: float) -> None:
        self._client = httpx.Client(timeout=timeout_seconds)

    def get_current_weather(self, location: str) -> WeatherResult:
        geocode_response = self._client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geocode_response.raise_for_status()
        geocode_payload = geocode_response.json()
        results = geocode_payload.get("results") or []
        if not results:
            raise ValueError(f"Unable to resolve weather location: {location}")

        best_match = results[0]
        forecast_response = self._client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": best_match["latitude"],
                "longitude": best_match["longitude"],
                "current": "temperature_2m,weather_code",
                "timezone": "auto",
            },
        )
        forecast_response.raise_for_status()
        forecast_payload = forecast_response.json()
        current = forecast_payload.get("current") or {}

        resolved_location = ", ".join(
            part
            for part in [best_match.get("name"), best_match.get("country_code")]
            if part
        )
        weather_code = int(current.get("weather_code", -1))
        return WeatherResult(
            temperature=round(float(current["temperature_2m"]), 1),
            summary=self._weather_codes.get(weather_code, "current conditions available"),
            location=resolved_location or location,
        )

    def close(self) -> None:
        self._client.close()
