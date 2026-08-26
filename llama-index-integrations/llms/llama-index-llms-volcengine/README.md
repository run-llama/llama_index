# LlamaIndex Llms Integration: Volcengine

This is the Volcengine Ark (Doubao) integration for LlamaIndex. Visit [Volcengine Ark](https://www.volcengine.com/docs/82379/) for information on how to get an API key and which models are supported.

## Installation

```bash
pip install llama-index-llms-volcengine
```

## Usage

Set the `ARK_API_KEY` environment variable or pass it directly to the constructor:

```python
from llama_index.llms.volcengine import Volcengine

llm = Volcengine(
    model="doubao-seed-2-1-pro-260628", api_key="your_api_key"
)
response = llm.complete("Hello")
print(response)
```
