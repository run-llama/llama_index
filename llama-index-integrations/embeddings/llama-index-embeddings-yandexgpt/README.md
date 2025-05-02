# LlamaIndex Embeddings Integration: YandexGPT

YandexGPT Embeddings provides text embedding service using the Yandex Cloud API. It supports both synchronous and asynchronous methods for embedding texts and documents(queries), making it flexible and suitable for various applications in natural language processing (NLP) and machine learning.

To learn more about Yandex Cloud API and embedding principles, visit https://yandex.cloud/en/docs/foundation-models/concepts/embeddings

## Installation

```bash
pip install llama-index-embeddings-yandexgpt
```

## Usage

```python
from llama_index.embeddings.yandexgpt import YandexGPTEmbedding
```

**Initialization Parameters:**

- `api_key`: The API key for Yandex Cloud. This key is required for authenticating requests.
- `folder_id`: The folder ID for Yandex Cloud.

```python
embeddings = YandexGPTEmbedding(
    api_key="your-api-key",
    folder_id="your-folder-id",
)
```

## Example

See the [example notebook](../../../docs/docs/examples/embeddings/yandexgpt.ipynb) for a detailed walkthrough of using YandexGPT embeddings with LlamaIndex.
