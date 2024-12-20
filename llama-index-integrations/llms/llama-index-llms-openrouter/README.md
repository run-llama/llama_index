# LlamaIndex Llms Integration: Openrouter

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-openrouter
!pip install llama-index
```

## Setup

### Initialize OpenRouter

You need to set either the environment variable `OPENROUTER_API_KEY` or pass your API key directly in the class constructor. Replace `<your-api-key>` with your actual API key:

```python
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage

llm = OpenRouter(
    api_key="<your-api-key>",
    max_tokens=256,
    context_window=4096,
    model="gryphe/mythomax-l2-13b",
)
```

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

To use a specific model, you can specify it during initialization. For example, to use Mistral's Mixtral model, you can set it like this:

```python
llm = OpenRouter(model="mistralai/mixtral-8x7b-instruct")
resp = llm.complete("Write a story about a dragon who can code in Rust")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/openrouter/
