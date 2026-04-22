# LlamaIndex LLM Integration: CometAPI

## Installation

To install the required packages, run:

```bash
pip install llama-index-llms-cometapi
```

## Setup

### Get API Key

1. Visit [CometAPI Console](https://api.cometapi.com/console/token)
2. Sign up for an account (If you don't already have a CometAPI account)
3. Generate your API key

### Initialize CometAPI

You can set the API key either as an environment variable `COMETAPI_API_KEY` or pass it directly:

```python
from llama_index.llms.cometapi import CometAPI

# Method 1: Using environment variable
# export COMETAPI_API_KEY="your-api-key"
llm = CometAPI(model="gpt-4o-mini")

# Method 2: Direct API key
llm = CometAPI(
    api_key="your-api-key",
    model="gpt-4o-mini",
    max_tokens=256,
    context_window=4096,
)
```

## Usage Examples

### Generate Chat Responses

```python
from llama_index.core.llms import ChatMessage

message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)
```

### Streaming Chat

```python
message = ChatMessage(role="user", content="Tell me a story")
resp = llm.stream_chat([message])
for r in resp:
    print(r.delta, end="")
```

### Text Completion

```python
resp = llm.complete("Tell me a joke")
print(resp)
```

### Streaming Completion

```python
resp = llm.stream_complete("Tell me a story")
for r in resp:
    print(r.delta, end="")
```

## Available Models

CometAPI supports various state-of-the-art models:

### GPT Series

- `gpt-5-chat-latest`
- `chatgpt-4o-latest`
- `gpt-5-mini`
- `gpt-4o-mini`
- `gpt-4.1-mini`

### Claude Series

- `claude-opus-4-1-20250805`
- `claude-sonnet-4-20250514`
- `claude-3-5-haiku-latest`

### Gemini Series

- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.0-flash`

### Others

- `deepseek-v3.1`
- `grok-4-0709`
- `qwen3-30b-a3b`

For complete list, visit: https://api.cometapi.com/pricing

## Model Configuration

```python
# Use different models
llm_claude = CometAPI(model="claude-3-5-haiku-latest")
llm_gemini = CometAPI(model="gemini-2.5-flash")
llm_deepseek = CometAPI(model="deepseek-v3.1")

response = llm_claude.complete("Explain quantum computing")
print(response)
```

## Documentation

- [CometAPI Website](https://www.cometapi.com/)
- [API Documentation](https://api.cometapi.com/doc)
- [Model Pricing](https://api.cometapi.com/pricing)
