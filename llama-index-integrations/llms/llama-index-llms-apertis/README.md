# LlamaIndex LLMs Integration: Apertis

Apertis provides a unified API gateway to access multiple LLM providers including OpenAI, Anthropic, Google, and more through an OpenAI-compatible interface.

## Installation

```bash
pip install llama-index-llms-apertis
```

## Supported Endpoints

Apertis supports multiple API formats:

| Endpoint               | Format                  | Description                             |
| ---------------------- | ----------------------- | --------------------------------------- |
| `/v1/chat/completions` | OpenAI Chat Completions | Default format used by this integration |
| `/v1/responses`        | OpenAI Responses        | OpenAI Responses format compatible      |
| `/v1/messages`         | Anthropic               | Anthropic format compatible             |

## Setup

### Get Your API Key

Obtain your API key from [Apertis API](https://api.apertis.ai/token).

### Initialize Apertis

You can set either the environment variable `APERTIS_API_KEY` or pass your API key directly in the class constructor:

```python
from llama_index.llms.apertis import Apertis
from llama_index.core.llms import ChatMessage

llm = Apertis(
    api_key="<your-api-key>",
    model="gpt-5.2",
)
```

Or using environment variables:

```bash
export APERTIS_API_KEY="<your-api-key>"
```

```python
from llama_index.llms.apertis import Apertis

llm = Apertis(model="gpt-5.2")
```

## Generate Chat Responses

Send a list of `ChatMessage` instances to generate a chat response:

```python
from llama_index.core.llms import ChatMessage

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

## Complete with Prompt

Generate completions with a prompt using the `complete` method:

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

## Supported Models

Apertis supports models from multiple providers:

| Provider  | Example Models                     |
| --------- | ---------------------------------- |
| OpenAI    | `gpt-5.2`, `gpt-5-mini-2025-08-07` |
| Anthropic | `claude-sonnet-4.5`                |
| Google    | `gemini-3-flash-preview`           |

### Using Different Models

```python
# Using Claude
llm = Apertis(
    api_key="<your-api-key>",
    model="claude-sonnet-4.5",
)

# Using Gemini
llm = Apertis(
    api_key="<your-api-key>",
    model="gemini-3-flash-preview",
)
```

## Configuration Options

| Parameter     | Description                | Default                     |
| ------------- | -------------------------- | --------------------------- |
| `api_key`     | Your Apertis API key       | `APERTIS_API_KEY` env var   |
| `api_base`    | API base URL               | `https://api.apertis.ai/v1` |
| `model`       | Model to use               | `gpt-5.2`                   |
| `temperature` | Sampling temperature       | `0.1`                       |
| `max_tokens`  | Maximum tokens to generate | `256`                       |
| `max_retries` | Maximum retry attempts     | `5`                         |

## Documentation

For more information, visit the [Apertis Documentation](https://docs.stima.tech).
