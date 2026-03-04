# LlamaIndex LLMs Integration: Helicone

## Installation

To install the required packages, run:

```bash
pip install llama-index-llms-helicone
pip install llama-index
```

## Setup

### Initialize Helicone

Set your Helicone API key via `HELICONE_API_KEY` (or pass directly). No provider API keys are needed when using the Helicone AI Gateway.

```python
from llama_index.llms.helicone import Helicone
from llama_index.core.llms import ChatMessage

llm = Helicone(
    api_key="<helicone-api-key>",  # or set HELICONE_API_KEY env var
    model="gpt-4o-mini",  # works across providers via gateway
)
```

## Generate Chat Responses

You can generate a chat response by sending a list of `ChatMessage` instances:

```python
message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)
```

### Streaming Responses

To stream responses, use the `stream_chat` method:

```python
message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message])
for r in resp:
    print(r.delta, end="")
```

### Complete with Prompt

You can also generate completions with a prompt using the `complete` method:

```python
resp = llm.complete("Tell me a joke")
print(resp)
```

### Streaming Completion

To stream completions, use the `stream_complete` method:

```python
resp = llm.stream_complete("Tell me a story in 250 words")
for r in resp:
    print(r.delta, end="")
```

## Model Configuration

To use a specific model, you can specify it during initialization. For example, to use Mistral's Mixtral model, you can set it like this:

```python
from llama_index.llms.helicone import Helicone

llm = Helicone(model="gpt-4o-mini")
resp = llm.complete("Write a story about a dragon who can code in Rust")
print(resp)
```

### Notes

- Default Helicone base URL is `https://ai-gateway.helicone.ai/v1`. Override with `api_base` or `HELICONE_API_BASE` if needed.
- Only `HELICONE_API_KEY` is required. The gateway routes to the correct provider based on the `model` string.
