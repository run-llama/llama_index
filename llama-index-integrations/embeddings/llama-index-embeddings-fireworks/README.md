# LlamaIndex Embeddings Integration: Fireworks

This integration provides support for [Fireworks AI](https://fireworks.ai/) embedding models with LlamaIndex.

Fireworks AI offers fast and efficient inference for embedding models. Sign up at [fireworks.ai](https://fireworks.ai/) to get an API key.

## Installation

```bash
pip install llama-index-embeddings-fireworks
```

## Usage

```python
from llama_index.embeddings.fireworks import FireworksEmbedding

# Initialize the embedding model
embed_model = FireworksEmbedding(
    api_key="your-api-key",  # or set FIREWORKS_API_KEY env var
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

# Get embedding for a single text
embedding = embed_model.get_text_embedding("Hello, world!")

# Get embeddings for multiple texts
embeddings = embed_model.get_text_embedding_batch(
    ["Hello, world!", "How are you?"]
)

# Get query embedding
query_embedding = embed_model.get_query_embedding("What is machine learning?")
```

## Configuration

| Parameter          | Description                                            | Default                          |
| ------------------ | ------------------------------------------------------ | -------------------------------- |
| `api_key`          | Fireworks API key (or set `FIREWORKS_API_KEY` env var) | None                             |
| `model_name`       | Embedding model to use                                 | `nomic-ai/nomic-embed-text-v1.5` |
| `dimensions`       | Output embedding dimensions (if supported by model)    | None                             |
| `embed_batch_size` | Batch size for embedding requests                      | 10                               |
| `timeout`          | Request timeout in seconds                             | 60.0                             |
| `max_retries`      | Maximum number of retries                              | 10                               |

## Environment Variables

- `FIREWORKS_API_KEY`: Your Fireworks API key
- `FIREWORKS_API_BASE`: Custom API base URL (optional)
