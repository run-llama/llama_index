# LlamaIndex Tools Integration: CAMB AI

This tool integrates [CAMB AI](https://camb.ai) audio and speech services with LlamaIndex.

## Features

- **Text-to-Speech**: Convert text to speech in 140+ languages
- **Translation**: Translate text between 140+ languages
- **Transcription**: Transcribe audio to text with speaker identification
- **Translated TTS**: Translate text and convert to speech in one step
- **Voice Cloning**: Clone a voice from a 2+ second audio sample
- **Voice Listing**: List all available voices
- **Text-to-Sound**: Generate sounds, music, or soundscapes from text descriptions
- **Audio Separation**: Separate vocals from background audio

## Installation

```bash
pip install llama-index-tools-camb
```

## Usage

```python
import os
from llama_index.tools.camb import CambToolSpec

os.environ["CAMB_API_KEY"] = "your-api-key"

camb = CambToolSpec()
tools = camb.to_tool_list()

# Use with an agent
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

agent = ReActAgent.from_tools(tools, llm=OpenAI())
agent.chat("Say hello in Spanish using text-to-speech")
```

## Configuration

```python
camb = CambToolSpec(
    api_key="your-api-key",  # or set CAMB_API_KEY env var
    base_url=None,  # optional custom API URL
    timeout=60.0,  # request timeout in seconds
    max_poll_attempts=60,  # max polling attempts for async tasks
    poll_interval=2.0,  # seconds between polls
)
```
