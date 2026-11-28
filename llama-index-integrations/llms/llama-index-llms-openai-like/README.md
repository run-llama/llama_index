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

## DaoXE (multi-model multi-protocol gateway)

[DaoXE](https://daoxe.com) exposes an OpenAI-compatible Chat Completions API at `https://daoxe.com/v1`.
Use an API key from the DaoXE dashboard and an **exact** model ID from your account catalog
(`GET /v1/models`). Do not hardcode a public model price list.

DaoXE also supports OpenAI Responses and Anthropic Messages for other clients; this package uses
the Chat Completions path via `OpenAILike`.

```python
import os
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model=os.environ["DAOXE_MODEL"],  # exact ID from your DaoXE account
    api_base="https://daoxe.com/v1",
    api_key=os.environ["DAOXE_API_KEY"],
    is_chat_model=True,
    is_function_calling_model=True,  # set False if your chosen model lacks tools
    context_window=128000,  # adjust to the model you select
)

print(llm.complete("Say hello in one short sentence."))
```

Examples: https://github.com/seven7763/DaoXE-AI

