# Heroku Managed Inference Embeddings

The `llama-index-embeddings-heroku` package contains LlamaIndex integrations for building applications with embedding models on Heroku's Managed Inference platform. This integration allows you to easily connect to and use embedding models deployed on Heroku's infrastructure.

## Installation

```shell
pip install llama-index
pip install llama-index-embeddings-heroku
```

## Setup

### 1. Create a Heroku App

First, create an app in Heroku:

```bash
heroku create $APP_NAME
```

### 2. Create and Attach Embedding Models

Create and attach an embedding model to your app:

```bash
heroku ai:models:create -a $APP_NAME cohere-embed-multilingual --as EMBEDDING
```

### 3. Export Configuration Variables

Export the required configuration variables:

```bash
export EMBEDDING_KEY=$(heroku config:get EMBEDDING_KEY -a $APP_NAME)
export EMBEDDING_MODEL_ID=$(heroku config:get EMBEDDING_MODEL_ID -a $APP_NAME)
export EMBEDDING_URL=$(heroku config:get EMBEDDING_URL -a $APP_NAME)
```

## Usage

### Basic Usage

```python
from llama_index.embeddings.heroku import HerokuEmbedding

# Initialize the Heroku Embedding
embedding_model = HerokuEmbedding()

# Get a single embedding
embedding = embedding_model.get_text_embedding("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")

# Get embeddings for multiple texts
texts = ["Hello", "world", "from", "Heroku"]
embeddings = embedding_model.get_text_embedding_batch(texts)
print(f"Number of embeddings: {len(embeddings)}")
```

### Using Parameters

You can also pass parameters directly:

```python
import os
from llama_index.embeddings.heroku import HerokuEmbedding

embedding_model = HerokuEmbedding(
    model=os.getenv("EMBEDDING_MODEL_ID", "cohere-embed-multilingual"),
    api_key=os.getenv("EMBEDDING_KEY", "your-inference-key"),
    base_url=os.getenv("EMBEDDING_URL", "https://us.inference.heroku.com"),
    timeout=60.0,
)

print(embedding_model.get_text_embedding("Hello Heroku!"))
```

### Async Usage

The integration also supports async operations:

```python
import asyncio
from llama_index.embeddings.heroku import HerokuEmbedding


async def get_embeddings_async():
    embedding_model = HerokuEmbedding()

    # Get async embeddings
    embedding = await embedding_model.aget_text_embedding("Hello, world!")
    embeddings = await embedding_model.aget_text_embedding_batch(
        ["Hello", "world"]
    )

    # Clean up
    await embedding_model.aclose()

    return embedding, embeddings


# Run async function
result = asyncio.run(get_embeddings_async())
print(result)
```

### Runnable Examples

See the `./examples` directory for more, runnable examples.

#### Running an Example

```bash
cd examples
uv run python basic_usage.py
```

### Integration with LlamaIndex

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.heroku import HerokuEmbedding
from llama_index.llms.heroku import Heroku
from llama_index.core import Document

# Set the LLM
llm = Heroku()
Settings.llm = llm

# Set the embedding model globally
Settings.embed_model = HerokuEmbedding()

# Create documents
documents = [
    Document(text="This is the first document"),
    Document(text="This is the second document"),
]

# Create a vector index
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine(
    llm=llm, response_mode="compact", similarity_top_k=5
)
response = query_engine.query("What documents do you have?")
print(response)
```

## Available Models

For a complete list of available embedding models, see the [Heroku Managed Inference documentation](https://devcenter.heroku.com/articles/heroku-inference#available-models).

## Error Handling

The integration includes proper error handling for common issues:

- Missing API key
- Invalid inference URL
- Missing model configuration
- Network errors
- HTTP errors

## Configuration Options

| Parameter          | Type  | Default                           | Description                          |
| ------------------ | ----- | --------------------------------- | ------------------------------------ |
| `model`            | str   | `os.getenv("EMBEDDING_MODEL_ID")` | The embedding model to use           |
| `api_key`          | str   | `os.getenv("EMBEDDING_KEY")`      | The API key for Heroku inference     |
| `base_url`         | str   | `os.getenv("EMBEDDING_URL")`      | The base URL for inference endpoints |
| `timeout`          | float | 60.0                              | Timeout for requests in seconds      |
| `embed_batch_size` | int   | 100                               | Batch size for embedding calls       |

## Environment Variables

| Variable             | Description                          |
| -------------------- | ------------------------------------ |
| `EMBEDDING_KEY`      | The API key for Heroku embedding     |
| `EMBEDDING_URL`      | The base URL for inference endpoints |
| `EMBEDDING_MODEL_ID` | The model ID to use                  |

## Testing

Run the test suite:

```bash
uv run -- pytest
```

Run with coverage:

```bash
uv run -- pytest --cov=llama_index tests/
```

## Additional Information

For more information about Heroku Managed Inference, visit the [official documentation](https://devcenter.heroku.com/articles/heroku-inference).

## License

This project is licensed under the MIT License.
