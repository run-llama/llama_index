"""Open Weather Map tool spec."""

from typing import Any, List

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class OpenWeatherMapToolSpec(BaseToolSpec):
    """Open Weather tool spec."""

    spec_functions = ["weather_at_location", "forecast_tommorrow_at_location"]

    def __init__(self, key: str, temp_units: str = "celsius") -> None:
        """Initialize with parameters."""
        try:
            from pyowm import OWM
        except ImportError:
            raise ImportError(
                "The OpenWeatherMap tool requires the pyowm package to be installed. "
                "Please install it using `pip install pyowm`."
            )

        self.key = key
        self.temp_units = temp_units
        self._owm = OWM(self.key)
        self._mgr = self._owm.weather_manager()

    def _format_temp(self, temperature: Any, temp_unit: str) -> str:
        return (
            f"  - Current: {temperature['temp']}{temp_unit}\n"
            f"  - High: {temperature['temp_max']}{temp_unit}\n"
            f"  - Low: {temperature['temp_min']}{temp_unit}\n"
            f"  - Feels like: {temperature['feels_like']}{temp_unit}"
        )

    def _format_weather(
        self, place: str, temp_str: str, w: Any, time_str: str = "now"
    ) -> str:
        """Format weather response from OpenWeatherMap.

        Function thanks to
        langchain/utilities/openweathermap.py
        """
        detailed_status = w.detailed_status
        wind = w.wind()
        humidity = w.humidity
        rain = w.rain
        heat_index = w.heat_index
        clouds = w.clouds

        return (
            f"In {place}, the weather for {time_str} is as follows:\n"
            f"Detailed status: {detailed_status}\n"
            f"Wind speed: {wind['speed']} m/s, direction: {wind['deg']}°\n"
            f"Humidity: {humidity}%\n"
            "Temperature: \n"
            f"{temp_str}\n"
            f"Rain: {rain}\n"
            f"Heat index: {heat_index!s}\n"
            f"Cloud cover: {clouds}%"
        )

    def weather_at_location(self, location: str) -> List[Document]:
        """
        Finds the current weather at a location.

        Args:
            place (str):
                The place to find the weather at.
                Should be a city name and country.
        """
        from pyowm.commons.exceptions import NotFoundError

        try:
            observation = self._mgr.weather_at_place(location)
        except NotFoundError:
            return [Document(text=f"Unable to find weather at {location}.")]

        w = observation.weather

        temperature = w.temperature(self.temp_units)
        temp_unit = "°C" if self.temp_units == "celsius" else "°F"
        temp_str = self._format_temp(temperature, temp_unit)

        weather_text = self._format_weather(location, temp_str, w)

        return [Document(text=weather_text, metadata={"weather from": location})]

    def forecast_tommorrow_at_location(self, location: str) -> List[Document]:
        """
        Finds the weather forecast for tomorrow at a location.

        Args:
            location (str):
                The location to find the weather tomorrow at.
                Should be a city name and country.
        """
        from pyowm.commons.exceptions import NotFoundError
        from pyowm.utils import timestamps

        try:
            forecast = self._mgr.forecast_at_place(location, "3h")
        except NotFoundError:
            return [Document(text=f"Unable to find weather at {location}.")]

        tomorrow = timestamps.tomorrow()
        w = forecast.get_weather_at(tomorrow)

        temperature = w.temperature(self.temp_units)
        temp_unit = "°C" if self.temp_units == "celsius" else "°F"
        temp_str = self._format_temp(temperature, temp_unit)

        weather_text = self._format_weather(location, temp_str, w, "tomorrow")

        return [
            Document(
                text=weather_text,
                metadata={
                    "weather from": location,
                    "forecast for": tomorrow.strftime("%Y-%m-%d"),
                },
            )
        ]
