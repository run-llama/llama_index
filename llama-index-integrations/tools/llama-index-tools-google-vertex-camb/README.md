# Google Vertex AI CAMB.AI MARS7 Tool

LlamaIndex integration for Google Vertex AI CAMB.AI MARS7 text-to-speech model with voice cloning capabilities.

## Installation

```bash
pip install llama-index-tools-google-vertex-camb
```

## Prerequisites

- Google Cloud project with Vertex AI API enabled
- Deployed MARS7 model endpoint (`publishers/cambai/models/mars7`)
- Service account credentials

## Usage

```python
from llama_index.tools.google_vertex_camb import GoogleVertexCambToolSpec

# Initialize
tool = GoogleVertexCambToolSpec(
    project_id="your-project-id",
    location="us-central1",
    endpoint_id="your-endpoint-id",
    credentials_path="/path/to/credentials.json",
)

# Basic text-to-speech
audio_file = tool.text_to_speech(text="Hello world!", language="en-us")

# Voice cloning
audio_file = tool.text_to_speech(
    text="This will sound like the reference voice",
    reference_audio_path="./reference.wav",
    reference_text="Reference transcription",
    language="en-us",
)
```

## Parameters

- `text` (str): Text to convert to speech
- `reference_audio_path` (str, optional): Reference audio for voice cloning
- `reference_text` (str, optional): Transcription of reference audio
- `language` (str): Language code (default: "en-us")
- `output_path` (str, optional): Output file path (default: "cambai_speech.flac")

---

## Supported Languages

- `de-de`: German (Germany)
- `en-gb`: English (United Kingdom)
- `en-us`: English (United States)
- `es-us`: Spanish (United States)
- `es-es`: Spanish (Spain)
- `fr-ca`: French (Canada)
- `fr-fr`: French (France)
- `ja-jp`: Japanese (Japan)
- `ko-kr`: Korean (South Korea)
- `zh-cn`: Chinese (Simplified, China)

---

## License

MIT
