# LlamaIndex Llms Integration: Tzafon

[Tzafon](https://tzafon.ai) provides fast, reliable AI inference with an OpenAI-compatible API. Access powerful language models through a simple, unified interface.

## Installation

```bash
pip install llama-index-llms-tzafon
```

## Usage

First, set your API key as an environment variable:

```bash
export TZAFON_API_KEY=sk_your_api_key_here
```

You can obtain an API key from the [Tzafon Console](https://console.tzafon.ai).

### Basic Usage

```python
from llama_index.llms.tzafon import Tzafon

# Initialize with default model (tzafon.sm-1)
llm = Tzafon()

# Or specify a model explicitly
llm = Tzafon(model="tzafon.sm-1")

# Generate a completion
response = llm.complete("Explain the importance of AI safety")
print(response)
```

### Chat Interface

```python
from llama_index.llms.tzafon import Tzafon
from llama_index.core.llms import ChatMessage

llm = Tzafon(model="tzafon.sm-1")

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is machine learning?"),
]

response = llm.chat(messages)
print(response)
```

### Streaming

```python
from llama_index.llms.tzafon import Tzafon

llm = Tzafon(model="tzafon.sm-1")

# Stream completions
for chunk in llm.stream_complete("Tell me a story about AI"):
    print(chunk.delta, end="", flush=True)
```

### Available Models

| Model                      | Description                                        |
| -------------------------- | -------------------------------------------------- |
| `tzafon.sm-1`              | Small, fast model for general tasks                |
| `tzafon.northstar.cua.sft` | Computer-use optimized model for automation agents |

### Configuration Options

```python
from llama_index.llms.tzafon import Tzafon

llm = Tzafon(
    model="tzafon.sm-1",  # Model to use
    api_key="sk_...",  # API key (or set TZAFON_API_KEY env var)
    temperature=0.7,  # Sampling temperature (0-2)
    max_tokens=1024,  # Maximum tokens to generate
    context_window=128000,  # Context window size
)
```

## Development

To create a development environment:

```bash
uv sync
```

## Testing

Run tests with:

```bash
uv run pytest tests
```

### Integration Tests

Integration tests require a valid API key:

```bash
export TZAFON_API_KEY=sk_your_key_here
uv run pytest tests
```

## Linting and Formatting

```bash
make format
make lint
```

## Resources

- [Tzafon Documentation](https://docs.tzafon.ai)
- [API Reference](https://docs.tzafon.ai/core-concepts/chat-completions)
- [Tzafon Console](https://console.tzafon.ai)
