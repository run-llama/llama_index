# LlamaIndex Llms Integration: DeepSeek

This is the DeepSeek integration for LlamaIndex. Visit [DeepSeek](https://api-docs.deepseek.com/) for information on how to get an API key and which models are supported.

## Installation

```bash
pip install llama-index-llms-deepseek
```

## Usage

```python
from llama_index.llms.deepseek import DeepSeek

llm = DeepSeek(model="deepseek-chat", api_key="your-api-key")
```
