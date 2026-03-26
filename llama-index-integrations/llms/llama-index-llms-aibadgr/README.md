# LlamaIndex LLMs Integration: AI Badgr

AI Badgr (Budget/Utility, OpenAI-compatible)

## Installation

To install the required packages, run:

```bash
pip install llama-index-llms-aibadgr
```

## Setup

### Initialize AI Badgr

You need to set either the environment variable `AIBADGR_API_KEY` or pass your API key directly in the class constructor. Replace `<your-api-key>` with your actual API key:

```python
from llama_index.llms.aibadgr import AIBadgr
from llama_index.core.llms import ChatMessage

llm = AIBadgr(
    api_key="<your-api-key>",
    model="premium",
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

AI Badgr supports tier-based model names for easy selection:

- **basic** - Budget tier model (maps to phi-3-mini)
- **normal** - Standard tier model (maps to mistral-7b)
- **premium** - Premium tier model (maps to llama3-8b-instruct, recommended default)

```python
# Using tier names (recommended)
llm = AIBadgr(model="premium", api_key="your_api_key")
resp = llm.complete("Write a story about a dragon who can code in Rust")
print(resp)
```

### Advanced: Power-User Model Names

You can also use specific model names directly:

```python
llm = AIBadgr(model="llama3-8b-instruct", api_key="your_api_key")
resp = llm.complete("Explain quantum computing")
print(resp)
```

OpenAI model names are accepted and mapped automatically.

## Environment Variables

You can configure AI Badgr using environment variables:

- `AIBADGR_API_KEY` - Your API key
- `AIBADGR_BASE_URL` - Custom base URL (default: https://aibadgr.com/api/v1)

```bash
export AIBADGR_API_KEY="your_api_key"
export AIBADGR_BASE_URL="https://aibadgr.com/api/v1"
```
