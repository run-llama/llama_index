# llama-index-vector-stores-azurepostgresql

Azure PostgreSQL Vector Store integration for [LlamaIndex](https://github.com/run-llama/llama_index).

This package provides an integration for using Azure Database for PostgreSQL as a vector store backend with LlamaIndex, supporting advanced vector search capabilities (including pgvector, DiskANN, and hybrid search).

## Features

- Store and query vector embeddings in Azure PostgreSQL
- Support for pgvector and DiskANN extensions
- Metadata filtering
- Seamless integration with LlamaIndex's core abstractions

## Installation

You can install the package and its dependencies using [uv](https://github.com/astral-sh/uv), pip, or poetry:

```bash
uv pip install .
# or
pip install .
# or
poetry install
```

**Dependencies:**

- `llama-index`
- `psycopg` (PostgreSQL driver)
- `azure-identity` (for Azure authentication)

## Usage Example

```python
import sys

sys.path.insert(0, "/path/to/llama-index-vector-stores-azurepostgresql")

from llama_index.vector_stores.azurepostgresql.base import AzurePGVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# Set up your Azure OpenAI and PostgreSQL connection details
llm = AzureOpenAI(...)
embed_model = AzureOpenAIEmbedding(...)

vector_store = AzurePGVectorStore.from_params(
    database="postgres",
    host="<your-host>.postgres.database.azure.com",
    port=5432,
    table_name="my_table",
    embed_dim=1536,
    pg_diskann_kwargs={
        "pg_diskann_operator_class": "vector_cosine_ops",
        "pg_diskann_max_neighbors": 32,
        "pg_diskann_l_value_ib": 100,
        "pg_diskann_l_value_is": 100,
        "pg_diskann_iterative_search": True,
    },
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
query_engine = index.as_query_engine()
response = query_engine.query("Your query here")
print(response)
```

## Development

- To run tests:
  ```bash
  make test
  ```
- To build the package:
  ```bash
  uv build
  ```

## License

This project is licensed under the terms of the Apache 2.0 license.
