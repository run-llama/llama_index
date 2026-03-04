# LlamaIndex x ElevenLabs integration

This package is an integration for ElevenLabs realtime conversation with agents.

To install the package, run:

```bash
python3 -m pip install llama-index-voice-agents-elevenlabs
```

And, if you want to run it, you can refer to the simple example down here (in this case, the audio input/output are the same as the local device you are running the script on):

```python
import os

from llama_index.voice_agents.elevenlabs import ElevenLabsVoiceAgent
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs


load_dotenv()
AGENT_ID = os.environ.get("AGENT_ID")
API_KEY = os.environ.get("ELEVENLABS_API_KEY")


def main():
    client = ElevenLabs(api_key=API_KEY)
    conversation = ElevenLabsVoiceAgent(
        client,
        AGENT_ID,
        requires_auth=bool(API_KEY),
    )
    conversation.start()

    while True:
        try:
            # GET MESSAGES IN llama-index ChatMessage FORMAT
            messages = conversation.export_messages()
            events = conversation.export_events()
            # GET AVERAGE LATENCY
            latency = conversation.average_latency
        except KeyboardInterrupt:
            conversation.interrupt()
            conversation.stop()
            print(f"Messages: {messages}")
            print(f"Events: {events}")
            print(f"Latency: {latency}")


if __name__ == "__main__":
    main()
```
