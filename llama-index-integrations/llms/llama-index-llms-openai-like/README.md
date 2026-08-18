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

## Governed OpenAI-compatible endpoints

You can also use `OpenAILike` with a governed OpenAI-compatible endpoint when you want LlamaIndex to keep owning retrieval, agents, tools, and workflows while a centralized control plane handles model access, policy, audit trails, quotas, routing, and cost reporting.

For example, [Tuning Engines](https://www.tuningengines.com/) exposes an OpenAI-compatible inference endpoint:

```python
import os

from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model=os.environ.get("TUNING_ENGINES_MODEL", "your-model-alias"),
    api_base="https://api.tuningengines.com/v1",
    api_key=os.environ["TUNING_ENGINES_API_KEY"],
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=True,
)
```

This pattern is useful for production RAG and agent applications that need centralized compliance, control, and cost visibility without changing their LlamaIndex application logic.
