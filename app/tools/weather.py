from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings
from app.resilience import AsyncCircuitBreaker, CircuitBreakerOpenError
from app.tools.base import BaseTool, ToolExecutionError


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Rain showers",
    95: "Thunderstorm",
}


class WeatherTool(BaseTool):
    name = "weather"
    description = "Get the current weather for a city or region."
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City, district, or place name to check weather for.",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
                "description": "Preferred temperature unit.",
            },
        },
        "required": ["location"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient,
        circuit_breaker: AsyncCircuitBreaker,
    ) -> None:
        self.settings = settings
        self.http_client = http_client
        self.circuit_breaker = circuit_breaker

    async def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        location = str(arguments.get("location", "")).strip()
        units = str(arguments.get("units", "celsius")).lower()
        if not location:
            raise ToolExecutionError("Weather location cannot be empty.")

        try:
            return await self.circuit_breaker.call(
                self._perform_weather_lookup,
                location,
                units,
            )
        except CircuitBreakerOpenError as exc:
            raise ToolExecutionError(
                f"Weather is temporarily unavailable because the circuit breaker is open. {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"Weather service request failed: {exc}") from exc

    async def _perform_weather_lookup(self, location: str, units: str) -> dict[str, Any]:
        geocode_response = await self.http_client.get(
            self.settings.geocoding_base_url,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geocode_response.raise_for_status()
        geocode_data = geocode_response.json()
        if not geocode_data.get("results"):
            raise ToolExecutionError(f"No weather location match found for '{location}'.")

        place = geocode_data["results"][0]
        temperature_unit = "fahrenheit" if units == "fahrenheit" else "celsius"
        weather_response = await self.http_client.get(
            self.settings.weather_base_url,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "timezone": "auto",
                "temperature_unit": temperature_unit,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
            },
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        current = weather_data.get("current")
        if not current:
            raise ToolExecutionError("Weather service returned no current conditions.")

        weather_code = current.get("weather_code")
        return {
            "status": "success",
            "location": {
                "name": place.get("name"),
                "country": place.get("country"),
                "admin1": place.get("admin1"),
                "latitude": place.get("latitude"),
                "longitude": place.get("longitude"),
            },
            "current": {
                "temperature": current.get("temperature_2m"),
                "apparent_temperature": current.get("apparent_temperature"),
                "relative_humidity": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation"),
                "wind_speed": current.get("wind_speed_10m"),
                "description": WEATHER_CODES.get(weather_code, f"Code {weather_code}"),
                "time": current.get("time"),
                "units": {"temperature": "deg F" if units == "fahrenheit" else "deg C"},
            },
        }
