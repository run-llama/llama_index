# ASI-1 Mini Integration for LlamaIndex

This package contains the LlamaIndex integration with [ASI-1 Mini](https://www.asi.ai/), a powerful language model designed for various natural language processing tasks.

ASI-1 Mini is the world's first Web3-native Large Language Model (LLM) developed by Fetch.ai Inc., a founding member of the Artificial Superintelligence Alliance. Unlike general-purpose LLMs, ASI-1 Mini is specifically designed and optimized for supporting complex agentic workflows.

With ASI-1 Mini, you can leverage these powerful capabilities:

- Advanced agentic reasoning with dynamic reasoning modes for complex tasks
- High performance on par with leading LLMs but with significantly lower hardware costs
- Specialized optimization for autonomous agent applications and multi-step tasks
- Seamless Web3 integration for secure and autonomous AI interactions

Want to learn more about ASI? Visit the [ASI website](https://asi1.ai) or [Fetch.ai](https://fetch.ai) for more information!

## Installation

```bash
pip install llama-index-llms-asi
```

## Usage

Here's an example of how to use the ASI integration with LlamaIndex:

```python
from llama_index.llms.asi import ASI

# Initialize the ASI LLM
llm = ASI(model="asi1-mini", api_key="your_api_key")

# Generate text
response = llm.complete("Tell me about artificial intelligence.")
print(response)

# Chat completion
from llama_index.core.llms import ChatMessage, MessageRole

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful AI assistant."
    ),
    ChatMessage(
        role=MessageRole.USER, content="Tell me about artificial intelligence."
    ),
]

response = llm.chat(messages)
print(response)
```

## Streaming Support

The ASI integration has different streaming implementations for completion and chat:

- **Streaming Completion**: ASI doesn't support streaming for completions (returns 404 error). Our implementation uses a fallback mechanism that returns the complete response as a single chunk.

- **Streaming Chat**: ASI supports streaming for chat, but with a unique format that includes:
  - Many empty content chunks during the "thinking" phase
  - Custom fields like `thought` and `init_thought` that contain intermediate reasoning
  - Actual content appearing later in the stream

Our implementation processes this format to filter out empty chunks and extract meaningful content, providing a clean streaming experience.

```python
# Streaming completion (falls back to regular completion)
for chunk in llm.stream_complete("Tell me about artificial intelligence."):
    print(chunk.text, end="", flush=True)

# Streaming chat (handles ASI's unique streaming format)
for chunk in llm.stream_chat(messages):
    if hasattr(chunk, "delta") and chunk.delta.strip():
        print(chunk.delta, end="", flush=True)
```

## Async Support

The ASI integration also supports async operations:

```python
# Async completion
response = await llm.acomplete("Tell me about artificial intelligence.")
print(response)

# Async chat
response = await llm.achat(messages)
print(response)

# Async streaming completion (falls back to regular completion)
async for chunk in llm.astream_complete(
    "Tell me about artificial intelligence."
):
    print(chunk.text, end="", flush=True)

# Async streaming chat (handles ASI's unique streaming format)
async for chunk in llm.astream_chat(messages):
    if hasattr(chunk, "delta") and chunk.delta.strip():
        print(chunk.delta, end="", flush=True)
```

## API Key

You need an API key to use ASI's API. You can provide it in two ways:

1. Pass it directly to the ASI constructor: `ASI(api_key="your_api_key")`
2. Set it as an environment variable: `export ASI_API_KEY="your_api_key"`

## Models

Currently, this integration supports the following models:

- `asi1-mini`: A powerful language model for various natural language processing tasks.

## Development

To create a development environment, install poetry then run:

```bash
poetry install --with dev
```

## Testing

To test the integration, first enter the poetry venv:

```bash
poetry shell
```

Then tests can be run with make

```bash
make test
```

### Integration tests

Integration tests will be skipped unless an API key is provided. API keys can be obtained from the Fetch.ai team.
Once created, store the API key in an environment variable and run tests

```bash
export ASI_API_KEY=<your key here>
make test
```

## Linting and Formatting

Linting and code formatting can be executed with make.

```bash
make format
make lint
```
