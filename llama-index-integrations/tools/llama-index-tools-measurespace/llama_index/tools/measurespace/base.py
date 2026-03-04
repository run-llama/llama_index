"""Measure Space tool spec."""

from typing import List, Dict

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MeasureSpaceToolSpec(BaseToolSpec):
    """Measure Space tool spec."""

    spec_functions = [
        "get_hourly_weather_forecast",
        "get_daily_weather_forecast",
        "get_daily_climate_forecast",
        "get_daily_air_quality_forecast",
        "get_latitude_longitude_from_location",
        "get_location_from_latitude_longitude",
    ]

    def __init__(self, api_keys: Dict[str, str], unit: str = "metric") -> None:
        """Initialize with parameters."""
        try:
            import measure_space_api as msa
        except ImportError:
            raise ImportError(
                "The Measure Space tool requires the measure-space-api package to be installed. "
                "Please install it using `pip install measure-space-api`."
            )

        self.api_keys = api_keys
        self.unit = unit
        self.msa = msa

    def _get_api_key(self, api_name: str):
        """
        Get API keys.

        Args:
            api_name (str): API service name

        """
        api_key = self.api_keys.get(api_name)
        if not api_key:
            raise ValueError(
                f"API key is required for {api_name} service. Please get your API key from measurespace.io/pricing."
            )

        return api_key

    def _format_output(self, wx: Dict) -> List[str]:
        """
        Format output to a list of string with the following format.

        ['total precipitation: 1 mm, wind speed: 10 m/s', 'total precipitation: 1 mm, wind speed: 10 m/s']

        Args:
            wx (Dict): API output in json format

        """
        wx_list = []
        for i in range(len(wx["time"])):
            tmp_list = []
            for key, value in wx.items():
                if key != "time":
                    a_name, a_unit = self.msa.get_metadata(key, self.unit)
                    tmp_list.append(f"{a_name}: {value[i]} {a_unit}")
            if tmp_list:
                wx_list.append(",".join(tmp_list))

        return wx_list

    def get_hourly_weather_forecast(self, location: str) -> List[Document]:
        """
        Get hourly weather forecast for given location.

        Args:
            location (str): location name

        """
        api_key = self._get_api_key("hourly_weather")
        geocoding_api_key = self._get_api_key("geocoding")
        params = {"variables": "tp, t2m, windSpeed, windDegree, r2"}
        wx = self.msa.get_hourly_weather(
            api_key,
            geocoding_api_key,
            location,
            params,
        )
        # Get variable metadata
        for x in ["latitude", "longitude"]:
            if x in wx:
                del wx[x]
        output = self._format_output(wx)
        documents = []
        for i in range(len(wx["time"])):
            documents.append(
                Document(
                    text=output[i],
                    metadata={
                        "Hourly weather for location": location,
                        "Date and time": wx["time"][i],
                    },
                )
            )

        return documents

    def get_daily_weather_forecast(self, location: str) -> List[Document]:
        """
        Get daily weather forecast for given location.

        Args:
            location (str): location name

        """
        api_key = self._get_api_key("daily_weather")
        geocoding_api_key = self._get_api_key("geocoding")
        params = {"variables": "tp, minT, maxT, meanwindSpeed, meanwindDegree, meanRH"}

        wx = self.msa.get_daily_weather(
            api_key,
            geocoding_api_key,
            location,
            params,
        )
        # Get variable metadata
        for x in ["latitude", "longitude"]:
            if x in wx:
                del wx[x]
        output = self._format_output(wx)
        documents = []

        for i in range(len(wx["time"])):
            documents.append(
                Document(
                    text=output[i],
                    metadata={
                        "Daily weather for location": location,
                        "Date": wx["time"][i],
                    },
                )
            )

        return documents

    def get_daily_climate_forecast(self, location: str) -> List[Document]:
        """
        Get hourly climate forecast for given location.

        Args:
            location (str): location name

        """
        api_key = self._get_api_key("daily_climate")
        geocoding_api_key = self._get_api_key("geocoding")
        params = {"variables": "t2m, tmin, tmax, sh2"}

        wx = self.msa.get_daily_climate(
            api_key,
            geocoding_api_key,
            location,
            params,
        )
        # Get variable metadata
        for x in ["latitude", "longitude"]:
            if x in wx:
                del wx[x]
        output = self._format_output(wx)
        documents = []

        for i in range(len(wx["time"])):
            documents.append(
                Document(
                    text=output[i],
                    metadata={
                        "Daily climate for location": location,
                        "Date": wx["time"][i],
                    },
                )
            )

        return documents

    def get_daily_air_quality_forecast(self, location: str) -> List[Document]:
        """
        Get daily air quality forecast for given location.

        Args:
            location (str): location name

        """
        api_key = self._get_api_key("daily_air_quality")
        geocoding_api_key = self._get_api_key("geocoding")
        params = {"variables": "AQI, maxPM10, maxPM25"}

        wx = self.msa.get_daily_air_quality(
            api_key,
            geocoding_api_key,
            location,
            params,
        )
        # Get variable metadata
        for x in ["latitude", "longitude"]:
            if x in wx:
                del wx[x]
        output = self._format_output(wx)
        documents = []

        for i in range(len(wx["time"])):
            documents.append(
                Document(
                    text=output[i],
                    metadata={
                        "Daily air quality for location": location,
                        "Date": wx["time"][i],
                    },
                )
            )

        return documents

    def get_latitude_longitude_from_location(self, location: str) -> List[Document]:
        """
        Get latitude and longitude from given location.

        Args:
            location (str): location name

        """
        api_key = self._get_api_key("geocoding")
        latitude, longitude = self.msa.get_lat_lon_from_city(
            api_key=api_key, location_name=location
        )

        return [
            Document(
                text=f"latitude: {latitude}, longitude: {longitude}",
                metadata={"Latitude and longitude for location": location},
            )
        ]

    def get_location_from_latitude_longitude(
        self, latitude: float, longitude: float
    ) -> List[Document]:
        """
        Get nearest location name from given latitude and longitude.

        Args:
            latitude (float): latitude
            longitude (float): longitude

        """
        api_key = self._get_api_key("geocoding")
        res = self.msa.get_city_from_lat_lon(api_key, latitude, longitude)

        return [
            Document(
                text=f"Location name: {res}",
                metadata="Nearest location for given longitude and latitude",
            )
        ]
