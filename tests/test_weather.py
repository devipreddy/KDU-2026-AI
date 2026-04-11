from multimodal_assistant.weather import MockWeatherClient


def test_mock_weather_client_is_deterministic() -> None:
    client = MockWeatherClient()

    first = client.get_current_weather("San Francisco, US")
    second = client.get_current_weather("San Francisco, US")

    assert first == second
    assert first.location == "San Francisco, US"
    assert isinstance(first.temperature, float)
