# LlamaIndex LLMs Integration: Crusoe

[Crusoe](https://crusoe.ai) provides a managed inference API powered by MemoryAlloy cluster-wide KV caching, delivering low-latency access to leading open models on renewable-energy AI infrastructure.

## Installation

```bash
pip install llama-index-llms-crusoe
```

Set your Crusoe API key as an environment variable:

```bash
export CRUSOE_API_KEY="your-api-key"
```

## Usage

### Basic Completion

```python
from llama_index.llms.crusoe import Crusoe

llm = Crusoe()
resp = llm.complete("Paul Graham is ")
print(resp)
```

### Basic Chat

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.crusoe import Crusoe

messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What is Crusoe Cloud?"),
]
resp = Crusoe().chat(messages)
print(resp)
```

### Streaming Completion

```python
from llama_index.llms.crusoe import Crusoe

llm = Crusoe()
for chunk in llm.stream_complete("Paul Graham is "):
    print(chunk.delta, end="")
```

### Streaming Chat

```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.crusoe import Crusoe

llm = Crusoe()
messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Tell me about renewable energy"),
]
for chunk in llm.stream_chat(messages):
    print(chunk.delta, end="")
```

### Model Configuration

```python
from llama_index.llms.crusoe import Crusoe

llm = Crusoe(model="deepseek-ai/DeepSeek-V3-0324")
resp = llm.complete("Explain quantum computing in simple terms.")
print(resp)
```

### Explicit API Key

```python
from llama_index.llms.crusoe import Crusoe

llm = Crusoe(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key="your-crusoe-api-key",
)
resp = llm.complete("Hello!")
print(resp)
```

### Custom Context Window (for new or unlisted models)

```python
from llama_index.llms.crusoe import Crusoe

llm = Crusoe(
    model="some/new-model",
    api_key="your-crusoe-api-key",
    context_window=65536,
    is_function_calling=True,
)
```

## Available Models

| Model | Context Window |
|---|---|
| `meta-llama/Llama-3.3-70B-Instruct` | 131,072 |
| `openai/gpt-oss-120b` | 131,072 |
| `deepseek-ai/DeepSeek-V3` | 163,840 |
| `deepseek-ai/DeepSeek-V3-0324` | 163,840 |
| `deepseek-ai/DeepSeek-R1-0528` | 163,840 |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | 262,144 |
| `google/gemma-3-12b-it` | 131,072 |
| `moonshotai/Kimi-K2-Thinking` | 262,144 |

For the full and up-to-date model list, see the [Crusoe Cloud console](https://console.crusoe.ai).
