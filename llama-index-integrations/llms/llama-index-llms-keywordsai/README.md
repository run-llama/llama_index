# LlamaIndex Llms Integration: KeywordsAI

## Installation

To install the required package, run:

```bash
%pip install llama-index-llms-keywordsai
```

## Setup

1. Set your KeywordsAI API key as an environment variable. You can replace `"sk-..."` with your actual API key:

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
```

## Basic Usage

### Generate Completions

To generate a completion for a prompt, use the `complete` method:

```python
from llama_index.llms.keywordsai import KeywordsAI

resp = KeywordsAI().complete("Paul Graham is ")
print(resp)
```

### Chat Responses

To send a chat message and receive a response, create a list of `ChatMessage` instances and use the `chat` method:

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality."
    ),
    ChatMessage(role="user", content="What is your name?"),
]
resp = KeywordsAI().chat(messages)
print(resp)
```

## Streaming Responses

### Stream Complete

To stream responses for a prompt, use the `stream_complete` method:

```python
from llama_index.llms.keywordsai import KeywordsAI

llm = KeywordsAI()
resp = llm.stream_complete("Paul Graham is ")
for r in resp:
    print(r.delta, end="")
```

### Stream Chat

To stream chat responses, use the `stream_chat` method:

```python
from llama_index.llms.keywordsai import KeywordsAI
from llama_index.core.llms import ChatMessage

llm = KeywordsAI()
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality."
    ),
    ChatMessage(role="user", content="What is your name?"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
```

## Configure Model

You can specify a particular model when creating the `KeywordsAI` instance:

```python
llm = KeywordsAI(model="gpt-3.5-turbo")
resp = llm.complete("Paul Graham is ")
print(resp)

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality."
    ),
    ChatMessage(role="user", content="What is your name?"),
]
resp = llm.chat(messages)
print(resp)
```

## Asynchronous Usage

You can also use asynchronous methods for completion:

```python
from llama_index.llms.keywordsai import KeywordsAI

llm = KeywordsAI(model="gpt-3.5-turbo")
resp = await llm.acomplete("Paul Graham is ")
print(resp)
```

## Set API Key at a Per-Instance Level

If desired, you can have separate LLM instances use different API keys:

```python
from llama_index.llms.keywordsai import KeywordsAI

llm = KeywordsAI(model="gpt-3.5-turbo", api_key="BAD_KEY")
resp = KeywordsAI().complete("Paul Graham is ")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/keywordsai/
