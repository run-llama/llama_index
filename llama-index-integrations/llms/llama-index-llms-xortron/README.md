# LlamaIndex LLMs Integration: Xortron

## Installation

```bash
pip install llama-index-llms-xortron
```

## Usage

### Basic Completion

```python
from llama_index.llms.xortron import Xortron

llm = Xortron(
    model="xortron-7b",
    base_url="http://localhost:8000",
)

response = llm.complete("What is the capital of France?")
print(response)
```

### Chat

```python
from llama_index.core.base.llms.types import ChatMessage, MessageRole

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
]

response = llm.chat(messages)
print(response.message.content)
```

### Streaming

```python
# Streaming completions
for chunk in llm.stream_complete("Tell me a story"):
    print(chunk.delta, end="", flush=True)

# Streaming chat
messages = [ChatMessage(role=MessageRole.USER, content="Tell me a joke")]
for chunk in llm.stream_chat(messages):
    print(chunk.delta, end="", flush=True)
```

### Async

```python
import asyncio

async def main():
    response = await llm.acomplete("What is 2 + 2?")
    print(response.text)

    # Async streaming
    gen = await llm.astream_complete("Count to five")
    async for chunk in gen:
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

### Error Handling

The client automatically retries transient errors (5xx status codes, timeouts, connection errors) with exponential backoff. Non-retryable errors (4xx) are raised immediately.

```python
import httpx

try:
    response = llm.complete("Hello")
except httpx.HTTPStatusError as e:
    print(f"HTTP error {e.response.status_code}: {e}")
except httpx.TimeoutException:
    print("Request timed out")
```

### Configuration

```python
llm = Xortron(
    model="xortron-13b",
    base_url="http://localhost:8000",
    temperature=0.5,
    max_tokens=1024,
    api_key="your-api-key",       # optional
    request_timeout=120.0,         # seconds (default: 60)
    max_retries=5,                 # retry transient errors (default: 3)
    additional_kwargs={"top_p": 0.9},
)
```

| Parameter | Default | Description |
|---|---|---|
| `model` | `"xortron-default"` | Model name on the Xortron server |
| `base_url` | `"http://localhost:8000"` | Server URL |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `256` | Maximum tokens to generate |
| `context_window` | `3900` | Context window size |
| `api_key` | `None` | Bearer token for authentication |
| `request_timeout` | `60.0` | HTTP request timeout in seconds |
| `max_retries` | `3` | Retries for transient errors (5xx, timeouts) |
| `additional_kwargs` | `{}` | Extra parameters passed to the API |
