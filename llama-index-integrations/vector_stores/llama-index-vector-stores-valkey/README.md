# LlamaIndex Vector_Stores Integration: Valkey

A LlamaIndex vector store using Valkey as the backend.

## Installation

```bash
pip install llama-index-vector-stores-valkey
```

## Usage

### Basic Example

```python
from llama_index.vector_stores.valkey import ValkeyVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Create vector store
vector_store = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
)

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
```

### Custom Schema with Metadata Filters

```python
from llama_index.vector_stores.valkey import ValkeyVectorStore
from llama_index.vector_stores.valkey.schema import ValkeyVectorStoreSchema
from glide_shared.commands.server_modules.ft_options.ft_create_options import (
    TagField,
    NumericField,
)

# Create custom schema with additional metadata fields
schema = ValkeyVectorStoreSchema()
schema.index.name = "my_index"
schema.index.prefix = "my_docs"
schema.add_fields(
    [
        TagField("category", "category"),
        NumericField("year", "year"),
    ]
)

vector_store = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    schema=schema,
    overwrite=True,
)
```

### Async Operations

```python
from llama_index.vector_stores.valkey import ValkeyVectorStore
from llama_index.core.schema import TextNode

vector_store = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
)

# Create index asynchronously
await vector_store.async_create_index()

# Add nodes
nodes = [TextNode(text="example", embedding=[0.1] * 1536)]
await vector_store.async_add(nodes)

# Query
from llama_index.core.vector_stores.types import VectorStoreQuery

query = VectorStoreQuery(
    query_embedding=[0.1] * 1536,
    similarity_top_k=5,
)
result = await vector_store.aquery(query)
```

## Features

- **Vector similarity search** using Valkey's vector search capabilities
- **Metadata filtering** with support for TAG, NUMERIC, and TEXT fields
- **Async support** for high-performance applications
- **Custom schemas** for flexible index configuration
- **Sync and async operations** for different use cases

## Exception Handling

All operations raise `ValkeyVectorStoreError` with descriptive messages:

```python
from llama_index.vector_stores.valkey import (
    ValkeyVectorStore,
    ValkeyVectorStoreError,
)

try:
    vector_store = ValkeyVectorStore(valkey_url="valkey://localhost:6379")
    vector_store.create_index()
    vector_store.add(nodes)
except ValkeyVectorStoreError as e:
    print(f"Operation failed: {e}")
```

## Requirements

- Valkey server with search module enabled
- Python >= 3.9

## More Information

- [Valkey Documentation](https://valkey.io/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
