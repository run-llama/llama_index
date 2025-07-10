# LlamaIndex Embeddings Integration: Baseten

This integration provides access to Baseten's embedding models through LlamaIndex using dedicated endpoints with OpenAI-compatible API format.

## Installation

```bash
pip install llama-index-embeddings-baseten
```

## Usage

### Basic Usage

```python
from llama_index.embeddings.baseten import BasetenEmbedding

# Using dedicated endpoint with OpenAI-compatible format
embed_model = BasetenEmbedding(
    model_id="03y7n6e3",  # Your specific Baseten model ID
    api_key="YOUR_BASETEN_API_KEY",
)

# Generate embeddings
embedding = embed_model.get_text_embedding("Hello, world!")
print(embedding)

# Batch embeddings
embeddings = embed_model.get_text_embedding_batch([
    "Hello, world!",
    "Goodbye, world!"
])
print(embeddings)
```

### Environment Variables

You can also set your API key as an environment variable:

```bash
export BASETEN_API_KEY="your-api-key"
```

```python
from llama_index.embeddings.baseten import BasetenEmbedding

# Will automatically pick up BASETEN_API_KEY from environment
embed_model = BasetenEmbedding(model_id="03y7n6e3")
```

### Integration with LlamaIndex

```python
from llama_index.core import Settings, VectorStoreIndex, Document

# Set as global embedding model
Settings.embed_model = BasetenEmbedding(
    model_id="03y7n6e3",
    api_key="YOUR_BASETEN_API_KEY"
)

# Use in vector store
documents = [Document(text="Your document text")]
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Your question")
```

## Features

- **OpenAI-Compatible API**: Inherits from OpenAIEmbedding for consistent behavior
- **Dedicated Endpoints**: Uses Baseten's dedicated endpoint URL format
- **Batch Processing**: Efficient batch embedding generation
- **Async Support**: Asynchronous embedding operations
- **Environment Variable Support**: Secure credential management
- **Inherited Functionality**: Leverages all OpenAIEmbedding features and optimizations

## API Format

Uses Baseten's dedicated endpoint with OpenAI-compatible API format:
```
POST https://model-{model_id}.api.baseten.co/environments/production/sync/v1/embeddings
{
    "model": "model_id",
    "input": "text string"
}
```

## Supported Models

Use your specific Baseten model ID (e.g., "03y7n6e3") for dedicated endpoints. You can deploy any OpenAI-compatible embedding model to Baseten and use it with this integration. 