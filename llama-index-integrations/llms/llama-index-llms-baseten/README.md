# LlamaIndex Llms Integration: Baseten

This integration allows you to use Baseten's hosted models with LlamaIndex.

## Installation

Install the required packages:

```bash
pip install llama-index-llms-baseten
pip install llama-index
```

## Model APIs vs. Dedicated Deployments

Baseten offers two main ways for inference.

1. Model APIs are public endpoints for popular open source models (GPT-OSS, Kimi K2, DeepSeek etc) where you can directly use a frontier model via slug e.g. `deepseek-ai/DeepSeek-V3-0324` and you will be charged on a per-token basis. You can find the list of supported models here: https://docs.baseten.co/development/model-apis/overview#supported-models.

2. Dedicated deployments are useful for serving custom models where you want to autoscale production workloads and have fine-grain configuration. You need to deploy a model in your Baseten dashboard and provide the 8 character model id like `abcd1234`.

By default, we set the `model_apis` parameter to `True`. If you want to use a dedicated deployment, you must set the `model_apis` parameter to `False` when instantiating the Baseten object.

## Usage

### Basic Usage (Dedicated Deployment)

To use Baseten models with LlamaIndex, first initialize the LLM:

```python
# Model APIs, you can find the model_slug here: https://docs.baseten.co/development/model-apis/overview#supported-models
llm = Baseten(
    model_id="MODEL_SLUG",
    api_key="YOUR_API_KEY",
    model_apis=True,  # Default, so not strictly necessary
)

# Dedicated Deployments, you can find the model_id by in the Baseten dashboard here: https://app.baseten.co/overview
llm = Baseten(
    model_id="MODEL_ID",
    api_key="YOUR_API_KEY",
    model_apis=False,
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

The async implementation uses webhooks to deliver results.

**Note: Async is only available for dedicated deployments and not for model APIs. `achat` is not supported because chat does not make sense for async operations.**

```python
async_llm = Baseten(
    model_id="your_model_id",
    api_key="your_api_key",
    webhook_endpoint="your_webhook_endpoint",
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
    headers={"Authorization": f"Api-Key {api_key}"},
)
print(resp.json())
```

For async operations, results are posted to your provided webhook endpoint. Your endpoint should validate the webhook signature and handle the results appropriately. The results are NOT stored by Baseten.

## Additional Resources

For more examples and detailed usage, check out the [Baseten Cookbook](https://docs.llamaindex.ai/en/stable/examples/llm/baseten/).

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/baseten.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
