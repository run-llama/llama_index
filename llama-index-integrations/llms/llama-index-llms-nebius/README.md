# LlamaIndex Llms Integration: [Nebius AI Studio](https://studio.nebius.ai/)

## Overview

Integrate with Nebius AI Studio API, which provides access to open-source state-of-the-art large language models (LLMs).

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

llm = NebiusLLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast")
```

#### Without environmental variables

```python
from llama_index.llms.nebius import NebiusLLM

llm = NebiusLLM(
    api_key="your_api_key", model="meta-llama/Meta-Llama-3.1-70B-Instruct-fast"
)
```

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
