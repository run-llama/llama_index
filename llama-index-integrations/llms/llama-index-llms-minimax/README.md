# LlamaIndex Llms Integration: MiniMax

This is the MiniMax integration for LlamaIndex. Visit [MiniMax](https://platform.minimax.io/docs/api-reference/text-openai-api) for information on how to get an API key and which models are supported.

## Installation

```bash
pip install llama-index-llms-minimax
```

## Usage

```python
from llama_index.llms.minimax import MiniMax

llm = MiniMax(model="MiniMax-M2.5", api_key="your-api-key")

response = llm.complete("Explain the importance of low latency LLMs")
print(response)
```

### Available Models

| Model                    | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| `MiniMax-M2.5`           | Peak Performance. Ultimate Value. Master the Complex. |
| `MiniMax-M2.5-highspeed` | Same performance, faster and more agile.              |

Both models support a 204,800-token context window.

### Environment Variables

You can set the `MINIMAX_API_KEY` environment variable instead of passing `api_key` directly:

```bash
export MINIMAX_API_KEY="your-api-key"
```

```python
from llama_index.llms.minimax import MiniMax

llm = MiniMax(model="MiniMax-M2.5")
```

### Custom Base URL

For users in mainland China, use the domestic API endpoint:

```python
llm = MiniMax(
    model="MiniMax-M2.5",
    api_base="https://api.minimaxi.com/v1",
)
```
