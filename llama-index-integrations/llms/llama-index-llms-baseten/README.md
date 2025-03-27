# LlamaIndex Llms Integration: Baseten

This integration allows you to use Baseten's hosted models with LlamaIndex.

## Installation

Install the required packages:

```bash
pip install llama-index-llms-baseten
pip install llama-index
```

## Usage

### Basic Usage

To use Baseten models with LlamaIndex, first initialize the LLM:

```python
from llama_index.llms.baseten import Baseten

llm = Baseten(
    model_id="your_model_id",  # e.g. "yqvr2lxw"
    api_key="your_api_key"
)
```

### Basic Completion

Generate a simple completion:

```python
response = llm.complete("Paul Graham is")
print(response.text)
```

### Chat Messages

Use chat-style interactions:

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
response = llm.chat(messages)
print(response)
```

### Streaming

Stream completions in real-time:

```python
# Streaming completion
response = llm.stream_complete("Paul Graham is")
for r in response:
    print(r.delta, end="")

# Streaming chat
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
response = llm.stream_chat(messages)
for r in response:
    print(r.delta, end="")
```

### Async Operations

Baseten supports async operations for long-running inference tasks. This is useful for:

- Tasks that may hit request timeouts
- Batch inference jobs
- Prioritizing certain requests

The async implementation uses webhooks to deliver results:

```python
async_llm = Baseten(
    model_id="your_model_id",
    api_key="your_api_key",
    webhook_endpoint="your_webhook_endpoint"
)
response = await async_llm.acomplete("Paul Graham is")
print(response)
```

To check the status of an async request:

```python
import requests

model_id = "your_model_id"
request_id = "your_request_id"
api_key = "your_api_key"

resp = requests.get(
    f"https://model-{model_id}.api.baseten.co/async_request/{request_id}",
    headers={"Authorization": f"Api-Key {api_key}"}
)
print(resp.json())
```

Note: For async operations, results are posted to your provided webhook endpoint. Your endpoint should validate the webhook signature and handle the results appropriately. The results are NOT stored by Baseten.

## Additional Resources

For more examples and detailed usage, check out the [Baseten Integration Cookbook](https://docs.llamaindex.ai/en/stable/examples/llm/baseten/).

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/baseten.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
