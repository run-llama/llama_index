# LlamaIndex Vector Store Integration: turbopuffer

[turbopuffer](https://turbopuffer.com) is a fast, cost-efficient vector database for search and retrieval. This integration supports dense vector search, BM25 full-text search, and hybrid search with Reciprocal Rank Fusion (RRF).

## Installation

```bash
pip install llama-index-vector-stores-turbopuffer
```

## Requirements

- Python >= 3.9, < 4.0
- A [turbopuffer](https://turbopuffer.com) account and API key
- `turbopuffer` >= 1.0.0, < 2.0.0
- `llama-index-core` >= 0.13.0

## Usage

```python
from turbopuffer import Turbopuffer
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.turbopuffer import TurbopufferVectorStore

# Initialize the client and namespace
tpuf = Turbopuffer(api_key="your-api-key", region="gcp-us-central1")
ns = tpuf.namespace("my-namespace")

# Create vector store and index
vector_store = TurbopufferVectorStore(namespace=ns)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

## Configuration

| Parameter         | Type        | Default             | Description                                           |
| ----------------- | ----------- | ------------------- | ----------------------------------------------------- |
| `namespace`       | `Namespace` | _required_          | A turbopuffer namespace handle                        |
| `distance_metric` | `str`       | `"cosine_distance"` | `"cosine_distance"` or `"euclidean_squared"`          |
| `batch_size`      | `int`       | `100`               | Batch size for write operations                       |
| `text_key`        | `str`       | `"text"`            | Attribute name for storing plain text (used for BM25) |

## Features

- **Dense vector search** — ANN similarity search using cosine distance or euclidean squared
- **BM25 full-text search** — native keyword search via `VectorStoreQueryMode.TEXT_SEARCH`
- **Hybrid search** — combine vector + BM25 with Reciprocal Rank Fusion via `VectorStoreQueryMode.HYBRID`
- **Metadata filtering** — supports all standard LlamaIndex filter operators (Eq, Gt, Lt, In, Contains, Glob, etc.)
- **Batch upserts** — configurable batch size for efficient writes
- **Delete by ID or filter** — remove nodes by ID list or metadata filters
