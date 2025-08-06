# LlamaIndex Llms Integration: Vercel AI Gateway

## Installation

To install the required packages, run:

```bash
%pip install llama-index-llms-vercel-ai-gateway
!pip install llama-index
```

## Setup

### Initialize Vercel AI Gateway

You need to set either the environment variable `VERCEL_AI_GATEWAY_API_KEY`, `VERCEL_OIDC_TOKEN`, or pass your API key directly in the class constructor. Replace `<your-api-key>` with your actual API key:

```python
from llama_index.llms.vercel_ai_gateway import VercelAIGateway
from llama_index.core.llms import ChatMessage

llm = VercelAIGateway(
    api_key="<your-api-key>",
    max_tokens=200000,
    context_window=64000,
    model="anthropic/claude-4-sonnet",
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

To use a specific model, you can specify it during initialization. For example, to use Anthropic's Claude 3 Sonnet model, you can set it like this:

```python
llm = VercelAIGateway(model="anthropic/claude-4-sonnet")
resp = llm.complete("Write a story about a dragon who can code in Rust")
print(resp)
```

### LLM Implementation example

https://docs.llamaindex.ai/en/stable/examples/llm/vercel-ai-gateway/
