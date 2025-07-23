# LlamaIndex x Gemini Live Integration

Integration between LlamaIndex and Google Gemini Live. Install the integration with:

```bash
pip install llama-index-voice-agents-gemini-live
```

And test it with the following minimal example:

```python
from llama_index.voice_agents.gemini_live import GeminiLiveVoiceAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.voice_agents import BaseVoiceAgentEvent
from llama_index.core.llms import ChatMessage, TextBlock
from typing import List
import random
import json


# use filter functions to export messages and events without your terminal being swamped by base64-encoded audio bytes :)
def filter_events(
    events: List[BaseVoiceAgentEvent],
) -> List[BaseVoiceAgentEvent]:
    evs = []
    for event in events:
        if not "audio" in event.type_t:
            evs.append(event)
    return evs


def filter_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    msgs = []
    for message in messages:
        msg = ChatMessage(role=message.role, blocks=[])
        for b in message.blocks:
            if isinstance(b, TextBlock):
                msg.blocks.append(b)
        if len(msg.blocks) > 0:
            msgs.append(msg)
    return msgs


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
    conversation = GeminiLiveVoiceAgent(tools=[weather_tool])

    await conversation.start()

    if conversation._quitflag:
        print("Events")
        print(conversation.export_events(filter=filter_events))
        print()
        print("Messages")
        print(conversation.export_messages(filter=filter_messages))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

Remember that:

- You have to either set `GOOGLE_API_KEY` as env variable or pass the `api_key` when initializing `GoogleGeminiVoiceAgent`
- You have to start the conversation
