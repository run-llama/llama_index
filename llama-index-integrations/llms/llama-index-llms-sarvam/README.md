# LlamaIndex Llms Integration: Servam

This is the Servam integration for LlamaIndex. Visit [Servam](https://docs.sarvam.ai/api-reference-docs/chat/completions) for information on how to get an API key and which models are supported.

## Installation

```bash
pip install llama-index-llms-servam
```

## Usage

```python
from llama_index.llms.servam import Servam

llm = Servam(model="servam-m", api_key="your-api-key")
```
