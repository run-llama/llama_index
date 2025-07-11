# LlamaIndex LLM Integration for Cloudflare AI Gateway

This package provides integration between LlamaIndex and Cloudflare AI Gateway, allowing you to use multiple AI models from different providers with automatic fallback.

## Features

- **Multiple Providers**: Use OpenAI, Anthropic, Mistral, Groq, and more
- **Automatic Fallback**: If one provider fails, automatically tries the next
- **Streaming Support**: Both chat and completion streaming
- **Async Support**: Full async/await support
- **Caching**: Built-in request caching with Cloudflare AI Gateway

## Installation

```bash
pip install llama-index-llms-cloudflare-ai-gateway
```

## Quick Start

```python
from llama_index.llms.cloudflare_ai_gateway import CloudflareAIGateway
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import ChatMessage

# Create LLM instances
openai_llm = OpenAI(model="gpt-4o-mini", api_key="your-openai-key")
anthropic_llm = Anthropic(
    model="claude-3-5-sonnet-20241022", api_key="your-anthropic-key"
)

# Create Cloudflare AI Gateway LLM
llm = CloudflareAIGateway(
    llms=[openai_llm, anthropic_llm],  # Try OpenAI first, then Anthropic
    account_id="your-cloudflare-account-id",
    gateway="your-ai-gateway-name",
    api_key="your-cloudflare-api-key",
)

# Use the LLM
messages = [ChatMessage(role="user", content="What is 2+2?")]
response = llm.chat(messages)
print(response.message.content)
```

## Supported Providers

- OpenAI
- Anthropic
- Mistral AI
- Groq
- DeepSeek
- Perplexity
- Replicate
- Grok
- Azure OpenAI

## Testing

```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export CLOUDFLARE_ACCOUNT_ID="your-id"
export CLOUDFLARE_API_KEY="your-key"
export CLOUDFLARE_GATEWAY="your-gateway"

# Run tests
uv run pytest tests/
```
