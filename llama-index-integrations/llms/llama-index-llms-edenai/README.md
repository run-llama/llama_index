# LlamaIndex Llms Integration: Eden AI

[Eden AI](https://www.edenai.co) gives you access to 500+ models from every major provider
(OpenAI, Anthropic, Google, Mistral, and more) through a single, OpenAI-compatible API and one
API key. Models are addressed in `provider/model` format, and an EU endpoint is available for
GDPR data residency.

## Installation

```bash
%pip install llama-index-llms-edenai
!pip install llama-index
```

## Setup

### Initialize Eden AI

Set your API key with the `EDENAI_API_KEY` environment variable or pass it directly. You can
create a key at [app.edenai.run](https://app.edenai.run).

```python
from llama_index.llms.edenai import EdenAI

llm = EdenAI(
    model="openai/gpt-4o-mini",
    api_key="<your-api-key>",  # or set EDENAI_API_KEY
)
```

Models use `provider/model` format, e.g. `openai/gpt-4o-mini`,
`anthropic/claude-sonnet-4-5`, `mistral/mistral-large-latest`. See the full list at
[app.edenai.run/models](https://app.edenai.run/models).

### EU data residency (GDPR)

To keep requests and data within the EU, use Eden AI's EU endpoint:

```python
llm = EdenAI(
    model="mistral/mistral-large-latest",
    api_base="https://api.eu.edenai.run/v3",  # or set EDENAI_API_BASE
)
```

## Usage

### Complete

```python
response = llm.complete("Hello World!")
print(str(response))
```

### Chat

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]
print(llm.chat(messages))
```

### Stream

```python
for chunk in llm.stream_complete("Tell me a story in 200 words"):
    print(chunk.delta, end="")
```
