# LlamaIndex LLM Integration: OpenLLM

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-openllm
!pip install llama-index
```

## Setup

### Initialize OpenLLM

First, import the necessary libraries and set up your `OpenLLM` instance. Replace `my-model`, `https://hostname.com/v1`, and `na` with your model name, API base URL, and API key, respectively:

```python
import os
from typing import List, Optional
from llama_index.llms.openllm import OpenLLM
from llama_index.core.llms import ChatMessage

llm = OpenLLM(
    model="my-model", api_base="https://hostname.com/v1", api_key="na"
)
```

## Generate Completions

To generate a completion, use the `complete` method:

```python
completion_response = llm.complete("To infinity, and")
print(completion_response)
```

### Stream Completions

You can also stream completions using the `stream_complete` method:

```python
async for it in llm.stream_complete(
    "The meaning of time is", max_new_tokens=128
):
    print(it, end="", flush=True)
```

## Chat Functionality

OpenLLM supports chat APIs, allowing you to handle conversation-like interactions. Here’s how to use it:

### Synchronous Chat

You can perform a synchronous chat by constructing a list of `ChatMessage` instances:

```python
from llama_index.core.llms import ChatMessage

chat_messages = [
    ChatMessage(role="system", content="You are acting as Ernest Hemmingway."),
    ChatMessage(role="user", content="Hi there!"),
    ChatMessage(role="assistant", content="Yes?"),
    ChatMessage(role="user", content="What is the meaning of life?"),
]

for it in llm.chat(chat_messages):
    print(it.message.content, flush=True, end="")
```

### Asynchronous Chat

To perform an asynchronous chat, use the `astream_chat` method:

```python
async for it in llm.astream_chat(chat_messages):
    print(it.message.content, flush=True, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/openllm/
