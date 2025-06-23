# LlamaIndex x OpenAI Realtime Integration

This package is an integration for the OpenAI Realtime Conversation service.

To install the package, run:

```bash
python3 -m pip install llama-index-voice-agents-openai
```

And, if you want to run it, you can refer to the simple example down here (in this case, the audio input/output are the same as the local device you are running the script on):

```python
import os
import signal
import time
import logging

from typing import List
from dotenv import load_dotenv
from llama_index.voice_agents.openai import OpenAIVoiceAgent
from llama_index.core.voice_agents import BaseVoiceAgentEvent
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.tools import FunctionTool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables from a .env file
load_dotenv()

quitFlag = False


# use filter functions to export messages and events without your terminal being swamped by base64-encoded audio bytes :)
def filter_events(
    events: List[BaseVoiceAgentEvent],
) -> List[BaseVoiceAgentEvent]:
    evs = []
    for event in events:
        if "audio" in event.type_t and "transcript" not in event.type_t:
            if "delta" in event.type_t:
                event.delta = ""
                evs.append(event)
        else:
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


def signal_handler(sig, frame):
    """Handle Ctrl+C and initiate graceful shutdown."""
    global quitFlag
    quitFlag = True


async def main():
    # this is just a mock tool
    def get_weather(location: str):
        """
        Get the weather for a location
        """
        return f"The weather in {location} is sunny, there are 32°C, wind is 3 km/h toward west, and humidity is at 34%."

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return
    tools = [
        FunctionTool.from_defaults(
            fn=get_weather,
            name="get_weather",
            description="Get the current weather...",
        )
    ]
    # use custom models and tools!
    conversation = OpenAIVoiceAgent(
        api_key=api_key,
        tools=tools,
        model="gpt-4o-mini-realtime-preview-2024-12-17",
    )

    signal.signal(
        signal.SIGINT,
        lambda sig, frame: signal_handler(sig, frame),
    )

    try:
        # customize the model with instructions and other arguments you can find here: https://platform.openai.com/docs/api-reference/realtime-client-events/session/update
        await conversation.start(
            instructions="You are a very helpful assistant. You can use the 'get_weather' tool to get weather information about a location: you just have to input the location to the tool. Please execute the tool whenever you are asked about the weather of a location.",
        )
        while not quitFlag:
            time.sleep(0.1)
        if quitFlag:
            await conversation.interrupt()
            await conversation.stop()

    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        await conversation.interrupt()
        await conversation.stop()

    finally:
        logging.info("Exiting main.")
        print("Messages:")
        print(conversation.export_messages(filter=filter_messages))
        print("Events:")
        print(conversation.export_events(filter=filter_events))
        await conversation.stop()  # Ensures cleanup if any error occurs


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    # remember that YOU have to start the conversation!
```
