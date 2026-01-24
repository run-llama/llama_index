# LlamaIndex Llms Integration: OVHcloud AI Endpoints

This integration allows you to use OVHcloud AI Endpoints with LlamaIndex. OVHcloud AI Endpoints provides OpenAI-compatible API endpoints for various models.

OVHcloud is a global player and the leading European cloud provider operating over 450,000 servers within 40 data centers across 4 continents to reach 1.6 million customers in over 140 countries. Our product AI Endpoints offers access to various models with sovereignty, data privacy and GDPR compliance.

## Installation

Install the required packages:

```bash
pip install llama-index llama-index-llms-ovhcloud
```

## API Key

OVHcloud AI Endpoints can be used in two ways:

1. **Free tier (with rate limits)**: You can use the API without an API key or with an empty string API key. This provides free access with rate limits.

2. **With API key**: For higher rate limits and production use, generate an API key from the OVHcloud manager:
   - Go to https://ovh.com/manager
   - Navigate to Public Cloud section
   - Go to AI & Machine Learning → AI Endpoints
   - Create an API key

## Usage

### Basic Usage

To use OVHcloud AI Endpoints with LlamaIndex, first initialize the LLM:

```python
from llama_index.llms.ovhcloud import OVHcloud

# Using with API key
llm = OVHcloud(
    model="gpt-oss-120b",
    api_key="YOUR_API_KEY",  # Or empty string for free tier with rate limits)
)
```

You can find available models in the [OVHcloud AI Endpoints catalog](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/).

### Basic Completion

Generate a simple completion:

```python
response = llm.complete("The capital of France is")
print(response.text)
```

### Chat Messages

Use chat-style interactions:

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What is the capital of France?"),
]
response = llm.chat(messages)
print(response)
```

### Streaming

Stream completions in real-time:

```python
# Streaming completion
response = llm.stream_complete("The capital of France is")
for r in response:
    print(r.delta, end="")

# Streaming chat
messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What is the capital of France?"),
]
response = llm.stream_chat(messages)
for r in response:
    print(r.delta, end="")
```

### Get Available Models

You can dynamically fetch available models:

```python
llm = OVHcloud(model="gpt-oss-120b")
available = llm.available_models  # List[Model] - fetched dynamically
model_ids = [model.id for model in available]
print(f"Available models: {model_ids}")
```

## Additional Resources

For more information about OVHcloud AI Endpoints, visit:

- [OVHcloud AI Endpoints Catalog](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/)
- [OVHcloud Manager](https://ovh.com/manager)
- [OVHcloud Help Centre](https://help.ovhcloud.com/csm/world-home?id=csm_index)
