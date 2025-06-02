# LlamaIndex Llms Integration: Featherless AI

## Installation

To install the required package, run:

```shell
pip install llama-index-llms-featherlessai
```

## Setup

1. Set your Featherless AI API key as an environment variable. Visit https://www.featherless.ai/ and sign up to get an API key.

```shell
import os

os.environ["FEATHERLESS_API_KEY"] = "you_api_key"
```

## Basic Usage

### Generate Completions

```python
from llama_index.llms.featherlessai import FeatherlessLLM

llm = FeatherlessLLM(model="Qwen/Qwen3-32B", api_key="your api key")

resp = llm.complete("Is 9.9 or 9.11 bigger?")
print(resp)
```

### Chat Responses

To send a chat message and receive a response, create a list of `ChatMessage` instances and use the `chat` method:

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)
print(resp)
```
