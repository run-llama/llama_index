# Measure Space Weather, Climate, Air Quality and Geocoding Tool

```bash
pip install llama-index-tools-measurespace
```

This tool connects to the [MeasureSpace](https://measurespace.io/documentation)'s API, using the [measure-space-api](https://pypi.org/project/measure-space-api/) Python package. You must initialize the tool with corresponding API keys from [MeasureSpace](https://measurespace.io/pricing) and [OpenAI](https://platform.openai.com/api-keys) (if you use OpenAI).

The tool has access to the following functions:

- hourly weather forecast (next 5 days)
- daily weather forecast (next 15 days)
- daily climate forecast (next 10 months)
- daily air quality forecast (next 5 days)
- get latitude and longitude from given city names
- get nearest city for given latitude and longitude

## Usage

Assume you have an `.env` file with the following content:

```env
GEOCODING_API_KEY=<your-geocoding-api-key>
HOURLY_WEATHER_API_KEY=<your-hourly-weather-api-key>
DAILY_WEATHER_API_KEY=R<your-daily-weather-api-key>
DAILY_CLIMATE_API_KEY=<your-daily-climate-api-key>
AIR_QUALITY_API_KEY=<your-air-quality-api-key>
OPENAI_API_KEY=<your-openai-api-key>
```

Note that you only need the API keys if you need the services.

```python
from llama_index.tools.measurespace import MeasureSpaceToolSpec
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv
import os

load_dotenv()

api_keys = {
    "hourly_weather": os.getenv("HOURLY_WEATHER_API_KEY"),
    "daily_weather": os.getenv("DAILY_WEATHER_API_KEY"),
    "daily_climate": os.getenv("DAILY_CLIMATE_API_KEY"),
    "air_quality": os.getenv("AIR_QUALITY_API_KEY"),
    "geocoding": os.getenv("GEOCODING_API_KEY"),
}

tool_spec = MeasureSpaceToolSpec(api_keys)
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("How's the temperature for New York in next 3 days?")
agent.chat("What's the latitude and longitude of New York?")

# get a list of tools
for tool in tool_spec.to_tool_list():
    print(tool.metadata.name)

# Use a specific tool
tool_spec.get_daily_weather_forecast("New York")
tool_spec.get_latitude_longitude_from_location("New York")
```
