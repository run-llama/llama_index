# LlamaIndex Vector Stores Integration: Vertex AI Vector Search

[Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) is a fully managed, highly scalable vector similarity search service on Google Cloud.

## Overview

This integration supports both Vertex AI Vector Search architectures:

- **v1.0** (default): Index + Endpoint architecture
- **v2.0** (opt-in): Collection-based architecture (simpler setup, more features)

The v2.0 support is opt-in and maintains 100% backward compatibility with existing v1.0 code.

## Installation

### Basic Installation (v1 only)

```bash
pip install llama-index-vector-stores-vertexaivectorsearch
```

### With v2 Support

```bash
pip install 'llama-index-vector-stores-vertexaivectorsearch[v2]'
```

## Quick Start

### v1.0 (Default)

```python
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

vector_store = VertexAIVectorStore(
    project_id="my-project",
    region="us-central1",
    index_id="projects/.../indexes/123",
    endpoint_id="projects/.../indexEndpoints/456",
    gcs_bucket_name="my-staging-bucket",  # Required for batch updates
)
```

### v2.0 (New - Simpler Setup)

```python
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

vector_store = VertexAIVectorStore(
    api_version="v2",  # Opt-in to v2
    project_id="my-project",
    region="us-central1",
    collection_id="my-collection"
    # No GCS bucket needed!
)
```

## Usage with LlamaIndex

```python
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

# Create vector store (v1 or v2)
vector_store = VertexAIVectorStore(
    api_version="v2",
    project_id="my-project",
    region="us-central1",
    collection_id="my-collection",
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index from documents
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is LlamaIndex?")
```

## v1 vs v2 Comparison

| Feature            | v1                 | v2              |
| ------------------ | ------------------ | --------------- |
| Required Resources | Index + Endpoint   | Collection only |
| GCS Bucket         | Required for batch | Not needed      |
| `clear()` method   | Not supported      | Supported       |
| Setup Complexity   | Higher             | Lower           |

## Parameters

### v1 Parameters

- `project_id`: Google Cloud project ID
- `region`: Google Cloud region
- `index_id`: Vertex AI index resource name
- `endpoint_id`: Vertex AI endpoint resource name
- `gcs_bucket_name`: GCS bucket for batch updates

### v2 Parameters

- `api_version`: Set to `"v2"`
- `project_id`: Google Cloud project ID
- `region`: Google Cloud region
- `collection_id`: Vertex AI collection name

## Documentation

- **v1.0 Usage**: See the [v1 example notebook](https://github.com/run-llama/llama_index/blob/main/docs/examples/vector_stores/VertexAIVectorSearchDemo.ipynb)
- **v2.0 Usage**: See the [v2 example notebook](https://github.com/run-llama/llama_index/blob/main/docs/examples/vector_stores/VertexAIVectorSearchV2Demo.ipynb)
- **v2.0 Migration**: See [V2_MIGRATION.md](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-vertexaivectorsearch/V2_MIGRATION.md) for detailed migration guide
- **API Reference**: See the [Google Cloud Documentation](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
