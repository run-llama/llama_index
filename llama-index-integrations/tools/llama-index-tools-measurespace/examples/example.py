from llama_index.tools.measurespace import MeasureSpaceToolSpec
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv
import os

load_dotenv()

api_keys = {
    'hourly_weather': os.getenv('HOURLY_WEATHER_API_KEY'),
    'daily_weather': os.getenv('DAILY_WEATHER_API_KEY'),
    'daily_climate': os.getenv('DAILY_CLIMATE_API_KEY'),
    'air_quality': os.getenv('AIR_QUALITY_API_KEY'),
    'geocoding': os.getenv('GEOCODING_API_KEY'),
}

tool_spec = MeasureSpaceToolSpec(api_keys)
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

agent.chat("How's the temperature for New York in next 3 days?")
agent.chat("What's the latitude and longitude of New York?")
