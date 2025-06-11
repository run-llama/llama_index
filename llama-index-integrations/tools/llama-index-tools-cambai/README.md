# LlamaIndex - CambAI Tool

This package contains the CambAI tool for LlamaIndex, which provides text-to-speech capabilities using the CambAI API.

## Installation

```bash
pip install llama-index-tools-cambai
```

## Usage

```python
from llama_index.tools.cambai import CambAIToolSpec

# Initialize the tool with your API key
tool = CambAIToolSpec(api_key="your_cambai_api_key")

# Get available voices
voices = tool.get_voices()
print(voices)

# Convert text to speech
output_path = tool.text_to_speech(
    text="Hello, this is a test of the CambAI text to speech API.",
    output_path="output.wav",
    voice_id=20303,  # Optional: specify a voice ID
)
print(f"Audio saved to: {output_path}")
```

## Requirements

- `cambai`: CambAI Python SDK
- `llama-index-core`: LlamaIndex core package

## License

This project is licensed under the MIT License - see the LICENSE file for details.
