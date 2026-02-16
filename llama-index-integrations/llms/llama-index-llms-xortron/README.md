# LlamaIndex LLMs Integration: Xortron

## Installation

```bash
pip install llama-index-llms-xortron
```

## Usage

### Basic Completion

```python
from llama_index.llms.xortron import Xortron

llm = Xortron(
    model="xortron-7b",
    base_url="http://localhost:8000",
)

response = llm.complete("What is the capital of France?")
print(response)
```

### Chat

```python
from llama_index.core.base.llms.types import ChatMessage, MessageRole

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=MessageRole.USER, content="What is the capital of France?"),
]

response = llm.chat(messages)
print(response.message.content)
```

### Streaming

```python
response = llm.stream_complete("Tell me a story")
for chunk in response:
    print(chunk.delta, end="", flush=True)
```

### Configuration

```python
llm = Xortron(
    model="xortron-13b",
    base_url="http://localhost:8000",
    temperature=0.5,
    max_tokens=1024,
    api_key="your-api-key",  # optional
    additional_kwargs={"top_p": 0.9},
)
```
