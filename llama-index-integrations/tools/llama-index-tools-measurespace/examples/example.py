from llama_index.tools.measurespace import MeasureSpaceToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
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
agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(
    await agent.run("How's the temperature for New York in next 3 days?")
)
print(
    await agent.run("What's the latitude and longitude of New York?")
)
