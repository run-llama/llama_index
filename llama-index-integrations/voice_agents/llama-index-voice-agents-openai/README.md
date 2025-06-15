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

from dotenv import load_dotenv
from llama_index.voice_agents.openai import OpenAIConversation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables from a .env file
load_dotenv()

quitFlag = False


def signal_handler(sig, frame, realtime_instance):
    """Handle Ctrl+C and initiate graceful shutdown."""
    realtime_instance.stop()
    global quitFlag
    quitFlag = True


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return

    conversation = OpenAIConversation(api_key=api_key)

    signal.signal(
        signal.SIGINT,
        lambda sig, frame: signal_handler(sig, frame, conversation),
    )

    try:
        conversation.start()
        while not quitFlag:
            time.sleep(0.1)

    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        conversation.stop()

    finally:
        logging.info("Exiting main.")
        print("Messages:")
        print(conversation.export_messages())
        print("Events:")
        print(conversation.export_events())
        conversation.stop()  # Ensures cleanup if any error occurs


if __name__ == "__main__":
    main()
```
