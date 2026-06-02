# LlamaIndex Llms Integration: MiniMax

This is the MiniMax integration for LlamaIndex. Visit [MiniMax](https://platform.minimax.io/docs/api-reference/text-openai-api) for information on how to get an API key and which models are supported.

## Installation

```bash
pip install llama-index-llms-minimax
```

## Usage

```python
from llama_index.llms.minimax import MiniMax

llm = MiniMax(model="MiniMax-M3", api_key="your-api-key")

response = llm.complete("Explain the importance of low latency LLMs")
print(response)
```

### Available Models

| Model                    | Context Window | Description                                                |
| ------------------------ | -------------- | ---------------------------------------------------------- |
| `MiniMax-M3`             | 524,288        | Latest flagship model, default                             |
| `MiniMax-M2.7`           | 204,800        | Previous-generation model, kept for backward compatibility |
| `MiniMax-M2.7-highspeed` | 204,800        | High-speed variant of M2.7 for low-latency scenarios       |

### Environment Variables

You can set the `MINIMAX_API_KEY` environment variable instead of passing `api_key` directly:

```bash
export MINIMAX_API_KEY="your-api-key"
```

```python
from llama_index.llms.minimax import MiniMax

llm = MiniMax(model="MiniMax-M3")
```

### Custom Base URL

For users in mainland China, use the domestic API endpoint:

```python
llm = MiniMax(
    model="MiniMax-M3",
    api_base="https://api.minimaxi.com/v1",
)
```
