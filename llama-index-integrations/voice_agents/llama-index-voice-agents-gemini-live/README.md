# LlamaIndex x Gemini Live Integration

Integration between LlamaIndex and Google Gemini Live. Install the integration with:

```bash
pip install llama-index-voice-agents-gemini-live
```

And test it with the following minimal example:

```python
from llama_index.voice_agents.gemini_live import GeminiLiveVoiceAgent
from llama_index.core.tools import FunctionTool
import random
import json


def get_weather(location: str) -> dict:
    """Fetch weather data for a given location."""
    return json.dumps(
        {
            "location": location,
            "temperature_c": round(random.uniform(15, 30), 1),
            "humidity_percent": random.randint(40, 90),
            "wind_speed_kmh": round(random.uniform(5, 25), 1),
            "precipitation_probability_percent": random.randint(0, 100),
        },
        indent=4,
    )


weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="Get the weather at a given location",
)


async def main():
    agent = GeminiLiveVoiceAgent(tools=[weather_tool])
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

Remember that:

- You have to either set `GOOGLE_API_KEY` as env variable or pass the `api_key` when initializing `GoogleGeminiVoiceAgent`
- You have to start the conversation
