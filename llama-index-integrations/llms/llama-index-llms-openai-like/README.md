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

### PZERO Example

[PZERO](https://pzero.studio) provides prepaid inference over OpenAI-compatible endpoints. Point `api_base` to `https://api.pzero.studio/v1` and ensure `is_chat_model=True`:

```python
import os
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="deepseek-v4-flash",
    api_base="https://api.pzero.studio/v1",
    api_key=os.environ["PZERO_API_KEY"],
    context_window=128000,
    is_chat_model=True,
)

response = llm.complete("Hello from PZERO")
print(str(response))
```

