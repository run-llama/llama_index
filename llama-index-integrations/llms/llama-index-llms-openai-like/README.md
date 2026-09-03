# LlamaIndex Llms Integration: Openai Like

`pip install llama-index-llms-openai-like`

This package is a thin wrapper around the OpenAI API. It is designed to be used with the OpenAI API, but can be used with any OpenAI-compatible API.

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

For example, TokenLab provides an OpenAI-compatible `/v1` endpoint:

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="claude-sonnet-5",
    api_base="https://api.tokenlab.sh/v1",
    api_key="your-tokenlab-api-key",
    context_window=200000,
    is_chat_model=True,
    is_function_calling_model=True,
)
```
