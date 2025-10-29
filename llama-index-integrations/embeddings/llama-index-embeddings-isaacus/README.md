# Isaacus Embeddings

The `llama-index-embeddings-isaacus` package contains LlamaIndex integrations for building applications with Isaacus' legal AI embedding models. This integration allows you to easily connect to and use state-of-the-art legal embeddings via the Isaacus API.

## Installation

```shell
pip install llama-index
pip install llama-index-embeddings-isaacus
```

## Setup

### 1. Create an Isaacus Account

Head to the [Isaacus Platform](https://platform.isaacus.com/accounts/signup/) to create a new account.

### 2. Add Payment Method and Get API Key

Once signed up, [add a payment method](https://platform.isaacus.com/billing/) to claim your [free credits](https://docs.isaacus.com/pricing/credits).

After adding a payment method, [create a new API key](https://platform.isaacus.com/users/api-keys/).

Make sure to keep your API key safe. You won't be able to see it again after you create it. But don't worry, you can always generate a new one.

### 3. Export Configuration Variables

Export your API key as an environment variable:

```bash
export ISAACUS_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from llama_index.embeddings.isaacus import IsaacusEmbedding

# Initialize the Isaacus Embedding model
# This uses the ISAACUS_API_KEY environment variable
embedding_model = IsaacusEmbedding()

# Get a single embedding
embedding = embedding_model.get_text_embedding("Legal document text here")
print(f"Embedding dimension: {len(embedding)}")

# Get embeddings for multiple texts
texts = ["Contract clause 1", "Contract clause 2", "Legal precedent"]
embeddings = embedding_model.get_text_embedding_batch(texts)
print(f"Number of embeddings: {len(embeddings)}")
```

### Using Parameters

You can also pass parameters directly and customize the embedding behavior:

```python
import os
from llama_index.embeddings.isaacus import IsaacusEmbedding

embedding_model = IsaacusEmbedding(
    model="kanon-2-embedder",  # Currently the only model available
    api_key=os.getenv("ISAACUS_API_KEY"),
    dimensions=1792,  # Optional: reduce dimensionality
    task="retrieval/document",  # Optimize for document retrieval
    timeout=60.0,
)

print(embedding_model.get_text_embedding("Legal text to embed"))
```

### Query vs Document Embeddings

Isaacus embeddings support task-specific optimization. Use `task="retrieval/query"` for search queries and `task="retrieval/document"` for documents:

```python
from llama_index.embeddings.isaacus import IsaacusEmbedding

# For documents
doc_embedder = IsaacusEmbedding(task="retrieval/document")
doc_embedding = doc_embedder.get_text_embedding("This is a legal document.")

# For queries (this is the default for get_query_embedding)
query_embedder = IsaacusEmbedding()
query_embedding = query_embedder.get_query_embedding(
    "Find documents about contracts"
)
```

### Async Usage

The integration also supports async operations:

```python
import asyncio
from llama_index.embeddings.isaacus import IsaacusEmbedding


async def get_embeddings_async():
    embedding_model = IsaacusEmbedding()

    # Get async embeddings
    embedding = await embedding_model.aget_text_embedding("Legal text here")
    embeddings = await embedding_model.aget_text_embedding_batch(
        ["Text 1", "Text 2"]
    )

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
from llama_index.embeddings.isaacus import IsaacusEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Document

# Set the LLM
llm = OpenAI()
Settings.llm = llm

# Set the Isaacus embedding model globally
Settings.embed_model = IsaacusEmbedding()

# Create documents
documents = [
    Document(text="This is a contract clause about payment terms."),
    Document(text="This is a contract clause about termination."),
]

# Create a vector index
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine(
    llm=llm, response_mode="compact", similarity_top_k=5
)
response = query_engine.query("What are the payment terms?")
print(response)
```

## Available Models

Currently, Isaacus offers the following embedding model:

- **kanon-2-embedder**: The world's most accurate legal embedding model on the [Massive Legal Embedding Benchmark (MLEB)](https://isaacus.com/blog/introducing-mleb) as of October 2025.

For more information about Isaacus models, see the [Isaacus documentation](https://docs.isaacus.com/models).

## Error Handling

The integration includes proper error handling for common issues:

- Missing API key
- Invalid API configuration
- Network errors
- API errors

## Configuration Options

| Parameter           | Type  | Default                        | Description                                          |
| ------------------- | ----- | ------------------------------ | ---------------------------------------------------- |
| `model`             | str   | "kanon-2-embedder"             | The embedding model to use                           |
| `api_key`           | str   | `os.getenv("ISAACUS_API_KEY")` | The API key for Isaacus                              |
| `base_url`          | str   | "https://api.isaacus.com/v1"   | The base URL for Isaacus API                         |
| `dimensions`        | int   | None (model default)           | Optional: reduce embedding dimensionality            |
| `task`              | str   | None                           | Task type: "retrieval/query" or "retrieval/document" |
| `overflow_strategy` | str   | "drop_end"                     | Strategy for handling overflow: "drop_end" or None   |
| `timeout`           | float | 60.0                           | Timeout for requests in seconds                      |
| `embed_batch_size`  | int   | 100                            | Batch size for embedding calls                       |

## Environment Variables

| Variable           | Description                             |
| ------------------ | --------------------------------------- |
| `ISAACUS_API_KEY`  | The API key for Isaacus (required)      |
| `ISAACUS_BASE_URL` | The base URL for Isaacus API (optional) |

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

For more information about Isaacus and its legal AI models:

- [Isaacus Documentation](https://docs.isaacus.com)
- [Kanon 2 Embedder Announcement](https://isaacus.com/blog/introducing-kanon-2-embedder)
- [Massive Legal Embedding Benchmark (MLEB)](https://isaacus.com/blog/introducing-mleb)
- [Isaacus Platform](https://platform.isaacus.com)

## License

This project is licensed under the MIT License.
