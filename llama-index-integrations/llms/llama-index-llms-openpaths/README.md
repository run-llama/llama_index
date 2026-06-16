# LlamaIndex Llms Integration: OpenPaths

[OpenPaths](https://openpaths.io) is an OpenAI-compatible model gateway. This
integration lets you call any model exposed by OpenPaths through the LlamaIndex
LLM interface.

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-openpaths
!pip install llama-index
```

## Setup

### Initialize OpenPaths

You need to set either the environment variable `OPENPATHS_API_KEY` or pass your API key directly in the class constructor. You can create a key on the [OpenPaths account page](https://openpaths.io/account). Replace `<your-api-key>` with your actual API key:

```python
from llama_index.llms.openpaths import OpenPaths
from llama_index.core.llms import ChatMessage

llm = OpenPaths(
    api_key="<your-api-key>",
    max_tokens=256,
    context_window=4096,
    model="openpaths/auto",
)
```

The default model is `openpaths/auto`, which lets OpenPaths route to a suitable
model automatically. Browse available models at
[https://openpaths.io/v1/models](https://openpaths.io/v1/models).

## Generate Chat Responses

You can generate a chat response by sending a list of `ChatMessage` instances:

```python
message = ChatMessage(role="user", content="Tell me a joke")
resp = llm.chat([message])
print(resp)
```

### Streaming Responses

To stream responses, use the `stream_chat` method:

```python
message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message])
for r in resp:
    print(r.delta, end="")
```

### Complete with Prompt

You can also generate completions with a prompt using the `complete` method:

```python
resp = llm.complete("Tell me a joke")
print(resp)
```

### Streaming Completion

To stream completions, use the `stream_complete` method:

```python
resp = llm.stream_complete("Tell me a story in 250 words")
for r in resp:
    print(r.delta, end="")
```

## Model Configuration

To use a specific model, you can specify it during initialization:

```python
llm = OpenPaths(model="openpaths/auto")
resp = llm.complete("Write a story about a dragon who can code in Rust")
print(resp)
```
