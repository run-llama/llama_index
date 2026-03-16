# LlamaIndex LLMs ModelsLab Integration

Provides [ModelsLab](https://modelslab.com) as an LLM provider for LlamaIndex — giving RAG pipelines, agents, and query engines access to uncensored Llama 3.1 models with 128K context windows.

## Installation

```bash
pip install llama-index-llms-modelslab
```

## Setup

Get your API key at [modelslab.com](https://modelslab.com), then:

```bash
export MODELSLAB_API_KEY="your-api-key"
```

## Usage

### Basic completion

```python
from llama_index.llms.modelslab import ModelsLabLLM

llm = ModelsLabLLM(model="llama-3.1-8b-uncensored")

resp = llm.complete("Explain how attention mechanisms work in transformers.")
print(resp)
```

### Chat

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="user",
        content="Write a Python function to merge two sorted lists.",
    ),
]
resp = llm.chat(messages)
print(resp)
```

### RAG pipeline

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.modelslab import ModelsLabLLM

Settings.llm = ModelsLabLLM(model="llama-3.1-70b-uncensored")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("Summarize the key findings.")
print(response)
```

### Streaming

```python
llm = ModelsLabLLM(model="llama-3.1-8b-uncensored")

for chunk in llm.stream_complete("Write a haiku about code:"):
    print(chunk.delta, end="", flush=True)
```

## Models

| Model                      | Context Window | Best for                               |
| -------------------------- | -------------- | -------------------------------------- |
| `llama-3.1-8b-uncensored`  | 128K           | Fast completions, most tasks (default) |
| `llama-3.1-70b-uncensored` | 128K           | Complex reasoning, high quality output |

## Configuration

```python
llm = ModelsLabLLM(
    model="llama-3.1-8b-uncensored",
    api_key="your-key",  # or MODELSLAB_API_KEY env var
    context_window=131072,  # 128K (default)
    temperature=0.7,  # sampling temperature
    max_tokens=2048,  # max output tokens
    is_chat_model=True,  # use chat endpoint (default)
)
```

## API Reference

- ModelsLab docs: https://docs.modelslab.com
- Uncensored chat endpoint: https://docs.modelslab.com/uncensored-chat
