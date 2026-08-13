# LlamaIndex LLMs Integration: [Nebius Token Factory](https://tokenfactory.nebius.com/)

## Overview

Integrate with the OpenAI-compatible Chat Completions API in Nebius Token Factory, which provides access to open-source large language models (LLMs).

## Installation

```bash
pip install llama-index-llms-nebius
```

## Usage

### Initialization

#### With environmental variables.

```.env
NEBIUS_API_KEY=your_api_key

```

```python
from llama_index.llms.nebius import NebiusLLM

llm = NebiusLLM(model="Qwen/Qwen3-30B-A3B-fast")
```

#### Without environmental variables

```python
from llama_index.llms.nebius import NebiusLLM

llm = NebiusLLM(api_key="your_api_key", model="Qwen/Qwen3-30B-A3B-fast")
```

The integration defaults to `https://api.tokenfactory.nebius.com/v1`. You can
set `api_base` explicitly to target a Dedicated Endpoint or another compatible
deployment. See the [Token Factory model list](https://docs.tokenfactory.nebius.com/models)
for currently available model IDs.

### Launching

#### Call `complete` with a prompt

```python
response = llm.complete("Amsterdam is the capital of ")
print(response)
```

#### Call `chat` with a list of messages

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful AI assistant."),
    ChatMessage(
        role="user",
        content="Write a poem about a smart AI robot named Wall-e.",
    ),
]
response = llm.chat(messages)
print(response)
```

#### Stream `complete`

```python
response = llm.stream_complete("Amsterdam is the capital of ")
for r in response:
    print(r.delta, end="")
```

#### Stream `chat`

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful AI assistant."),
    ChatMessage(
        role="user",
        content="Write a poem about a smart AI robot named Wall-e.",
    ),
]
response = llm.stream_chat(messages)
for r in response:
    print(r.delta, end="")
```
