# LlamaIndex Llms Integration: Openai Like

`pip install llama-index-llms-openai-like`

This package is a thin wrapper around the OpenAI API. It is designed to be used
with the OpenAI API, but can be used with any OpenAI-compatible API, including
Foundry Local.

## Usage

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="model-name",
    api_base="http://localhost:1234/v1",
    api_key="fake",
    # Explicitly set the context window to match the model's context window
    context_window=128000,
    # Controls whether the model uses chat or completion endpoint
    is_chat_model=True,
    # Controls whether the model supports function calling
    is_function_calling_model=False,
)
```

## Foundry Local

Install `llama-index-llms-openai-like` and the Foundry Local SDK for your platform:

```bash
# Windows
pip install llama-index-llms-openai-like foundry-local-sdk-winml

# macOS/Linux
pip install llama-index-llms-openai-like foundry-local-sdk
```

```python
from foundry_local_sdk import Configuration, FoundryLocalManager
from llama_index.llms.openai_like import OpenAILike

FoundryLocalManager.initialize(Configuration(app_name="llamaindex-foundry-local"))
manager = FoundryLocalManager.instance

model = manager.catalog.get_model("qwen2.5-0.5b")
model.download()
model.load()
manager.start_web_service()

llm = OpenAILike(
    model=model.id,
    api_base=f"{manager.urls[0].rstrip('/')}/v1",
    api_key="fake",
    context_window=32768,
    is_chat_model=True,
    is_function_calling_model=False,
    timeout=120.0,
)
```

Foundry Local itself is SDK-first and does not need to run as a web server unless
you are connecting it to an OpenAI-compatible HTTP client like `OpenAILike`. This
example uses the optional local `/v1` endpoint only for that compatibility layer.
Foundry Local 1.2 also exposes the OpenAI Responses API from the same `/v1`
endpoint, but `OpenAILike` continues to use the chat-completions surface.
