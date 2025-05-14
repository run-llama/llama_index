# LlamaIndex Llms Integration: Ollama

## Installation

To install the required package, run:

```bash
pip install llama-index-llms-ollama
```

## Setup

1. Follow the [Ollama README](https://ollama.com) to set up and run a local Ollama instance.
2. When the Ollama app is running on your local machine, it will serve all of your local models on `localhost:11434`.
3. Select your model when creating the `Ollama` instance by specifying `model=":"`.
4. You can increase the default timeout (30 seconds) by setting `Ollama(..., request_timeout=300.0)`.
5. If you set `llm = Ollama(..., model="<model family>")` without a version, it will automatically look for the latest version.

## Usage

### Initialize Ollama

```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
```

### Generate Completions

To generate a text completion for a prompt, use the `complete` method:

```python
resp = llm.complete("Who is Paul Graham?")
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
resp = llm.chat(messages)
print(resp)
```

### Streaming Responses

#### Stream Complete

To stream responses for a prompt, use the `stream_complete` method:

```python
response = llm.stream_complete("Who is Paul Graham?")
for r in response:
    print(r.delta, end="")
```

#### Stream Chat

To stream chat responses, use the `stream_chat` method:

```python
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

### JSON Mode

Ollama supports a JSON mode to ensure all responses are valid JSON, which is useful for tools that need to parse structured outputs:

```python
llm = Ollama(model="llama3.1:latest", request_timeout=120.0, json_mode=True)
response = llm.complete(
    "Who is Paul Graham? Output as a structured JSON object."
)
print(str(response))
```

### Structured Outputs

You can attach a Pydantic class to the LLM to ensure structured outputs:

```python
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools import FunctionTool


class Song(BaseModel):
    """A song with name and artist."""

    name: str
    artist: str


llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
sllm = llm.as_structured_llm(Song)

response = sllm.chat([ChatMessage(role="user", content="Name a random song!")])
print(
    response.message.content
)  # e.g., {"name": "Yesterday", "artist": "The Beatles"}
```

### Asynchronous Chat

You can also use asynchronous chat:

```python
response = await sllm.achat(
    [ChatMessage(role="user", content="Name a random song!")]
)
print(response.message.content)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/ollama/
