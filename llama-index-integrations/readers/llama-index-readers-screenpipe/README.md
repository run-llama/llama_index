# LlamaIndex Readers Integration: Screenpipe

Reads screen capture (OCR) and audio transcription data from a local
[Screenpipe](https://github.com/mediar-ai/screenpipe) instance.

Screenpipe is a 24/7 local screen & mic recording tool that captures screen
content via OCR and audio transcriptions via Whisper, storing everything in a
local SQLite database and exposing a REST API.

## Installation

```bash
pip install llama-index-readers-screenpipe
```

## Usage

Make sure Screenpipe is running locally (default: `http://localhost:3030`).

```python
from llama_index.readers.screenpipe import ScreenpipeReader
from llama_index.core import VectorStoreIndex

reader = ScreenpipeReader()

# Load recent screen and audio data
documents = reader.load_data(content_type="all", limit=50)

# Load only audio transcriptions with a query filter
from datetime import datetime, timedelta

documents = reader.load_data(
    content_type="audio",
    query="meeting notes",
    start_time=datetime.now() - timedelta(hours=24),
)

# Build an index and query
index = VectorStoreIndex.from_documents(documents)
engine = index.as_query_engine()
response = engine.query("What did I discuss in my last meeting?")
```
