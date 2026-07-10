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
| `MiniMax-M3`             | 1,000,000      | Latest flagship model, default                             |
| `MiniMax-M2.7`           | 204,800        | Previous-generation model, kept for backward compatibility |
| `MiniMax-M2.7-highspeed` | 204,800        | High-speed variant of M2.7 for low-latency scenarios       |
| `MiniMax-M2.5`           | 204,800        | Existing model retained for backward compatibility         |
| `MiniMax-M2.5-highspeed` | 204,800        | High-speed variant retained for backward compatibility     |

### Environment Variables

You can set the `MINIMAX_API_KEY` environment variable instead of passing `api_key` directly:

```bash
export MINIMAX_API_KEY="your-api-key"
```

```python
from llama_index.llms.minimax import MiniMax

llm = MiniMax(model="MiniMax-M3")
```

### Regional Base URLs

The `MiniMax` class uses the OpenAI-compatible API. MiniMax publishes matching OpenAI-compatible and Anthropic-compatible endpoints in both regions:

| Region         | OpenAI-compatible base URL    | Anthropic-compatible base URL           | Documentation                        |
| -------------- | ----------------------------- | --------------------------------------- | ------------------------------------ |
| Global         | `https://api.minimax.io/v1`   | `https://api.minimax.io/anthropic/v1`   | `https://platform.minimax.io/docs`   |
| Mainland China | `https://api.minimaxi.com/v1` | `https://api.minimaxi.com/anthropic/v1` | `https://platform.minimaxi.com/docs` |

The global OpenAI-compatible URL is the default. To use the mainland China OpenAI-compatible endpoint, pass it as `api_base`:

```python
llm = MiniMax(
    model="MiniMax-M3",
    api_base="https://api.minimaxi.com/v1",
)
```
