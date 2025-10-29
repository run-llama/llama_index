# LlamaIndex Vector Stores Integration: Couchbase

This package provides Couchbase vector store integrations for LlamaIndex, offering multiple implementation options for vector similarity search based on Couchbase Server's native vector indexing capabilities.

## Installation

```bash
pip install llama-index-vector-stores-couchbase
```

## Available Vector Store Classes

### CouchbaseSearchVectorStore

Implements [Search Vector Indexes](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html) using Couchbase Full-Text Search (FTS) with vector search capabilities. Ideal for hybrid searches combining vector, full-text, and geospatial searches.

### CouchbaseQueryVectorStore (Recommended)

Implements both [Hyperscale Vector Indexes](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html) and [Composite Vector Indexes](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html) using Couchbase Query Service with SQL++ and vector search functions. Supports:

- **Hyperscale Vector Indexes**: Purpose-built for pure vector searches at massive scale with minimal memory footprint
- **Composite Vector Indexes**: Best for combining vector similarity with scalar filters that exclude large portions of the dataset

Can scale to billions of documents. Requires Couchbase Server 8.0+.

### CouchbaseVectorStore (Deprecated)

> **Note:** `CouchbaseVectorStore` has been deprecated in version 0.4.0. Please use `CouchbaseSearchVectorStore` instead.

## Requirements

- Python >= 3.9, < 4.0
- Couchbase Server 7.6+ for Search Vector Indexes
- Couchbase Server 8.0+ for Hyperscale and Composite Vector Indexes
- couchbase >= 4.5.0

## Basic Usage

### Using CouchbaseSearchVectorStore (Search Vector Indexes)

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

### Using CouchbaseQueryVectorStore (Hyperscale & Composite Vector Indexes)

```python
from llama_index.vector_stores.couchbase import (
    CouchbaseQueryVectorStore,
    QueryVectorSearchType,
    QueryVectorSearchSimilarity,
)

# Initialize Query Service-based vector store
# Works with both Hyperscale Vector Indexes (pure vector search)
# and Composite Vector Indexes (vector + scalar filters)
vector_store = CouchbaseQueryVectorStore(
    cluster=cluster,
    bucket_name="my_bucket",
    scope_name="my_scope",
    collection_name="my_collection",
    search_type=QueryVectorSearchType.ANN,  # or QueryVectorSearchType.KNN
    similarity=QueryVectorSearchSimilarity.COSINE,  # Can also use string: "cosine", "euclidean", "dot_product"
    nprobes=10,  # Optional: number of probes for ANN search (only for ANN)
    text_key="text",
    embedding_key="embedding",
    metadata_key="metadata",
)
```

## Configuration Options

### Search Types

The `QueryVectorSearchType` enum defines the type of vector search to perform:

- `QueryVectorSearchType.ANN` - Approximate Nearest Neighbor (recommended for large datasets)
- `QueryVectorSearchType.KNN` - K-Nearest Neighbor (exact search)

### Similarity Metrics

The `QueryVectorSearchSimilarity` enum provides various distance metrics:

- `QueryVectorSearchSimilarity.COSINE` - Cosine similarity (range: -1 to 1)
- `QueryVectorSearchSimilarity.DOT` - Dot product similarity
- `QueryVectorSearchSimilarity.L2` or `EUCLIDEAN` - Euclidean distance
- `QueryVectorSearchSimilarity.L2_SQUARED` or `EUCLIDEAN_SQUARED` - Squared Euclidean distance

You can also use lowercase strings: `"cosine"`, `"dot_product"`, `"euclidean"`, etc.

## Features

- **Multiple Index Types**: Support for all three Couchbase vector index types:
  - Hyperscale Vector Indexes (Query Service-based, 8.0+)
  - Composite Vector Indexes (Query Service-based, 8.0+)
  - Search Vector Indexes (FTS-based, 7.6+)
- **Flexible Similarity Metrics**: Multiple distance metrics including:
  - COSINE (Cosine similarity)
  - DOT (Dot product)
  - L2 / EUCLIDEAN (Euclidean distance)
  - L2_SQUARED / EUCLIDEAN_SQUARED (Squared Euclidean distance)
- **Metadata Filtering**: Advanced filtering capabilities using LlamaIndex MetadataFilters
- **Batch Operations**: Efficient batch insertion with configurable batch sizes
- **High Performance**: ANN and KNN search support for efficient nearest neighbor queries
- **Massive Scalability**: Hyperscale and Composite indexes can scale to billions of documents

## Implementation Details

### Query Service-Based Vector Indexes (`CouchbaseQueryVectorStore`)

`CouchbaseQueryVectorStore` supports both **Hyperscale Vector Indexes** and **Composite Vector Indexes**, which use the Couchbase Query Service with SQL++ queries and vector search functions.

#### Hyperscale Vector Indexes

Purpose-built for pure vector searches at massive scale:

**When to Use:**

- Pure vector similarity searches without complex scalar filtering
- Content discovery, recommendations, reverse image search
- Chatbot context matching (e.g., RAG workflows)
- Anomaly detection in IoT sensor networks
- Datasets from tens of millions to billions of documents

**Key Characteristics:**

- Optimized specifically for vector searches
- Higher accuracy at lower quantizations
- Low memory footprint (most index data on disk)
- Best TCO for huge datasets
- Excellent for concurrent updates and searches
- Scalar values and vectors compared simultaneously

#### Composite Vector Indexes

Combine a Global Secondary Index (GSI) with vector search functions:

**When to Use:**

- Searches that combine vector similarity with scalar filters
- When scalar filters can exclude large portions (>20%) of the dataset
- Applications requiring compliance-based restrictions on results
- Content recommendations, job searches, supply chain management
- Datasets from tens of millions to billions of documents

**Key Characteristics:**

- Scalar filters are applied _before_ vector search, reducing vectors to compare
- Efficient when scalar values have low selectivity (exclude <20% of dataset)
- Can exclude nearest neighbors based on scalar values (useful for compliance)
- Can scale to billions of documents

#### Search Types (Both Hyperscale & Composite)

- **ANN (Approximate Nearest Neighbor)**: Faster approximate search with configurable `nprobes` parameter for accuracy/speed tradeoff
- **KNN (K-Nearest Neighbor)**: Exact nearest neighbor search for maximum accuracy

### Search Vector Indexes (`CouchbaseSearchVectorStore`)

Search Vector Indexes combine Full-Text Search (FTS) with vector search capabilities:

**When to Use:**

- Hybrid searches combining vector, full-text, and geospatial searches
- Applications like e-commerce product search, travel recommendations, or real estate searches
- Datasets up to tens of millions of documents

**Key Characteristics:**

- Combines semantic search with keyword and geospatial searches in a single query
- Supports both scoped and global indexes
- Ideal for multi-modal search scenarios

### Metadata Filtering

Both implementations support metadata filtering:

- Filter by document attributes using standard LlamaIndex `MetadataFilters`
- Supports operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `IN`, `NIN`
- Combine filters with `AND`/`OR` conditions

### Choosing the Right Index Type

The same `CouchbaseQueryVectorStore` class works with both Hyperscale and Composite Vector Indexes. The choice of which underlying index type to use is determined by the index you create on your Couchbase collection.

| Feature             | Hyperscale (via QueryVectorStore)    | Composite (via QueryVectorStore) | Search (via SearchVectorStore)     |
| ------------------- | ------------------------------------ | -------------------------------- | ---------------------------------- |
| **Index Type**      | Hyperscale Vector Index              | Composite Vector Index           | Search Vector Index                |
| **Best For**        | Pure vector searches                 | Vector + scalar filters          | Vector + full-text + geospatial    |
| **Available Since** | Couchbase Server 8.0                 | Couchbase Server 8.0             | Couchbase Server 7.6               |
| **Scalar Handling** | Compared with vectors simultaneously | Pre-filters before vector search | Searches in parallel               |
| **Use Cases**       | Content discovery, RAG, image search | Job search, compliance filtering | E-commerce, travel recommendations |

For more information, refer to: [Couchbase Vector Search Documentation](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html)

## License

MIT
