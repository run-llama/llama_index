# LlamaIndex Embeddings Integration: Netmind

## Installation

To install the required package, run:

```shell
pip install llama-index-llms-netmind
```

## Setup

1. Set your Netmind API key as an environment variable. Visit https://www.netmind.ai/ and sign up to get an API key.

```shell
import os

os.environ["NETMIND_API_KEY"] = "you_api_key"
```

## Basic Usage

```python
from llama_index.embeddings.netmind import NetmindEmbedding

embed_model = NetmindEmbedding(model_name="BAAI/bge-m3", api_key="")
embeddings = embed_model.get_text_embedding("hello world")
print(len(embeddings))
```
