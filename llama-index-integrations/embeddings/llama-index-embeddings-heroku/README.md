# LlamaIndex Embeddings Integration: Heroku

This package provides a LlamaIndex embedding integration for the Heroku Inference API, enabling you to use Cohere embedding models through Heroku's managed inference service.

## Installation

```bash
pip install llama-index-embeddings-heroku
```

## Usage

```python
from llama_index.embeddings.heroku import HerokuEmbedding

# Initialize the embedding model
embed_model = HerokuEmbedding(
    api_key="your-heroku-inference-key",
    model="cohere-embed-multilingual",  # default
)

# Get embeddings for a single text
embedding = embed_model.get_text_embedding("Hello, world!")

# Get embeddings for multiple texts
embeddings = embed_model.get_text_embedding_batch(["Hello", "World"])

# Use with LlamaIndex VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | Your Heroku Inference API key |
| `model` | str | `"cohere-embed-multilingual"` | The embedding model to use |
| `base_url` | str | `"https://us.inference.heroku.com"` | Heroku Inference API base URL |
| `embed_batch_size` | int | `96` | Maximum batch size for embedding requests |
| `timeout` | float | `60.0` | Request timeout in seconds |

## Available Models

The Heroku Inference API supports the following embedding models:

- `cohere-embed-multilingual` (1024 dimensions) - Recommended for most use cases
- `cohere-embed-english` (1024 dimensions) - Optimized for English text

## Environment Variables

You can also configure the API key using an environment variable:

```bash
export HEROKU_INFERENCE_KEY="your-api-key"
```

```python
import os
from llama_index.embeddings.heroku import HerokuEmbedding

embed_model = HerokuEmbedding(
    api_key=os.environ["HEROKU_INFERENCE_KEY"]
)
```

## License

MIT
