# llama-index-vector-stores-azurepostgresql

Azure PostgreSQL Vector Store integration for [LlamaIndex](https://github.com/run-llama/llama_index).

This package provides an integration for using Azure Database for PostgreSQL as a vector store backend with LlamaIndex, supporting advanced vector search capabilities (including pgvector, DiskANN, and hybrid search).

## Features

- Store and query vector embeddings in Azure PostgreSQL
- Support for pgvector and DiskANN extensions
- Metadata filtering
- Seamless integration with LlamaIndex's core abstractions

## Installation

You can install the package using pip:

```bash
pip install llama-index-vector-stores-azurepostgres
```

**Dependencies:**

- `llama-index`
- `psycopg` (PostgreSQL driver)
- `azure-identity` (for Azure authentication)

## Usage Example

```python
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.azure_postgres import AzurePGVectorStore
from llama_index.vector_stores.azure_postgres.common import (
    AzurePGConnectionPool,
    BasicAuth,
    ConnectionInfo,
    DiskANN,
    SSLMode,
)

# Set up the models LlamaIndex should use.
Settings.llm = AzureOpenAI(...)
Settings.embed_model = AzureOpenAIEmbedding(...)

# Describe how to reach your Azure Database for PostgreSQL instance. Use a
# TokenCredential from azure-identity in place of BasicAuth for Entra ID auth.
connection_info = ConnectionInfo(
    host="<your-host>.postgres.database.azure.com",
    dbname="postgres",
    port=5432,
    sslmode=SSLMode.require,
    credentials=BasicAuth(username="<user>", password="<password>"),
)
connection_pool = AzurePGConnectionPool(azure_conn_info=connection_info)

# Choose the vector index to create. DiskANN is shown here; HNSW and IVFFlat
# are also exported from the same module.
embedding_index = DiskANN(
    op_class="vector_cosine_ops",
    max_neighbors=32,
    l_value_ib=100,
    l_value_is=100,
)

vector_store = AzurePGVectorStore.from_params(
    connection_pool=connection_pool,
    schema_name="public",
    table_name="llamaindex_vectors",
    embed_dim=1536,
    embedding_index=embedding_index,
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
