# LlamaIndex Llms Integration: Neutrino

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-neutrino
!pip install llama-index
```

## Setup

### Create Neutrino API Key

You can create an API key by visiting [platform.neutrinoapp.com](https://platform.neutrinoapp.com). Once you have the API key, set it as an environment variable:

```python
import os

os.environ["NEUTRINO_API_KEY"] = "<your-neutrino-api-key>"
```

## Using Your Router

A router is a collection of LLMs that you can route queries to. You can create a router in the Neutrino dashboard or use the default router, which includes all supported models. You can treat a router as a single LLM.

### Initialize Neutrino

Create an instance of the Neutrino model:

```python
from llama_index.llms.neutrino import Neutrino

llm = Neutrino(
    # api_key="<your-neutrino-api-key>",
    # router="<your-router-id>"  # Use 'default' for the default router
)
```

### Generate Completions

To generate a text completion for a prompt, use the `complete` method:

```python
response = llm.complete("In short, a Neutrino is")
print(f"Optimal model: {response.raw['model']}")
print(response)
```

### Chat Responses

To send a chat message and receive a response, create a `ChatMessage` and use the `chat` method:

```python
from llama_index.core.llms import ChatMessage

message = ChatMessage(
    role="user",
    content="Explain the difference between statically typed and dynamically typed languages.",
)

resp = llm.chat([message])
print(f"Optimal model: {resp.raw['model']}")
print(resp)
```

### Streaming Responses

To stream responses for a chat message, use the `stream_chat` method:

```python
message = ChatMessage(
    role="user", content="What is the approximate population of Mexico?"
)

resp = llm.stream_chat([message])
for i, r in enumerate(resp):
    if i == 0:
        print(f"Optimal model: {r.raw['model']}")
    print(r.delta, end="")
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/neutrino/
