# LlamaIndex Vector Stores Integration: Couchbase

This package provides Couchbase vector store integrations for LlamaIndex, offering multiple implementation options for vector similarity search.

## Installation

```bash
pip install llama-index-vector-stores-couchbase
```

## Available Vector Store Classes

### CouchbaseSearchVectorStore

Uses Couchbase Full-Text Search (FTS) with vector search capabilities.

### CouchbaseQueryVectorStore (Recommended)

Uses Couchbase Global Secondary Index (GSI) with BHIVE vector search support for high-performance ANN operations.

### CouchbaseVectorStore (Deprecated)

> **Note:** `CouchbaseVectorStore` has been deprecated in version 0.4.0. Please use `CouchbaseSearchVectorStore` instead.

## Requirements

- Python >= 3.9, < 4.0
- Couchbase Server with vector search capabilities
- couchbase >= 4.2.0, < 5

## Basic Usage

### Using CouchbaseSearchVectorStore (FTS-based)

```python
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator

# Connect to Couchbase
auth = PasswordAuthenticator("username", "password")
cluster = Cluster("couchbase://localhost", auth)

# Initialize vector store
vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name="my_bucket",
    scope_name="my_scope",
    collection_name="my_collection",
    index_name="my_vector_index",
    text_key="text",
    embedding_key="embedding",
    metadata_key="metadata",
    scoped_index=True,
)
```

### Using CouchbaseQueryVectorStore (GSI-based)

```python
from llama_index.vector_stores.couchbase import (
    CouchbaseQueryVectorStore,
    QueryVectorSearchType,
)

# Initialize GSI-based vector store
vector_store = CouchbaseQueryVectorStore(
    cluster=cluster,
    bucket_name="my_bucket",
    scope_name="my_scope",
    collection_name="my_collection",
    search_type=QueryVectorSearchType.ANN,  # or QueryVectorSearchType.KNN
    similarity="cosine",  # or "euclidean", "dot_product"
    nprobes=10,  # Optional: number of probes for ANN search
    text_key="text",
    embedding_key="embedding",
    metadata_key="metadata",
)
```

## Features

- **Multiple Search Types**: Support for both GSI-based and FTS vector search
- **Flexible Similarity Metrics**: Cosine, Euclidean, and dot product similarities
- **Metadata Filtering**: Advanced filtering capabilities using LlamaIndex MetadataFilters
- **Batch Operations**: Efficient batch insertion with configurable batch sizes
- **High Performance**: BHIVE index support for approximate nearest neighbor (ANN) search
- **Scoped Indexes**: Support for both scoped and global search indexes in FTS-based vector search

## Search Types

### ANN (Approximate Nearest Neighbor)

- Uses BHIVE indexes for high-performance approximate search
- Configurable nprobes parameter for accuracy/speed tradeoff
- Recommended for large-scale deployments

### KNN (K-Nearest Neighbor)

- Exact nearest neighbor search
- Higher accuracy but potentially slower for large datasets
- Good for smaller datasets or when exact results are required

For more information, refer to: [Couchbase Vector Search Concepts](https://preview.docs-test.couchbase.com/docs-server-DOC-12565_vector_search_concepts/server/current/vector-index/use-vector-indexes.html)

## License

MIT
