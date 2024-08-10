# LlamaIndex Embeddings Integration: GigaChat

GigaChat Embedding provides a way to generate embeddings for text and documents using the GigaChat API within the llama_index library.

To learn more about GigaChat and embedding principles, visit https://developers.sber.ru/docs/ru/gigachat/api/embeddings?tool=api

## Installation

```bash
pip install gigachat
pip install llama-index-embeddings-gigachat
```

## Usage

```python
from llama_index.embeddings.gigachat import GigaChatEmbedding
```

**Initialization Parameters:**

- `auth_data`: GigaChat authentication data.
- `scope`: The scope of your GigaChat API access. Use "GIGACHAT_API_PERS" for personal use or "GIGACHAT_API_CORP" for corporate use.

```python
embeddings = GigaChatEmbedding(
    auth_data="YOUR_AUTH_DATA",
    scope="GIGACHAT_API_CORP",
)
```

## Example

See the [example notebook](https://github.com/run-llama/llama_index/tree/main/docs/docs/examples/embeddings/gigachat.ipynb) for a detailed walkthrough of using GigaChat embeddings with LlamaIndex.
