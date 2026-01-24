# LlamaIndex Vector Stores Integration: Couchbase

This package provides Couchbase vector store integrations for LlamaIndex, offering multiple implementation options for vector similarity search based on Couchbase Server's native vector indexing capabilities.

## Installation

```bash
pip install llama-index-vector-stores-couchbase
```

## Available Vector Store Classes

### CouchbaseSearchVectorStore

Implements [Search Vector Indexes](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html) using Couchbase Search Service with vector search capabilities. Ideal for hybrid searches combining vector, full-text, and geospatial searches.

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
  - Hyperscale Vector Indexes (8.0+)
  - Composite Vector Indexes (8.0+)
  - Search Vector Indexes (7.6+)
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

Combines a standard Global Secondary index (GSI) with a single vector column:

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

Search Vector Indexes use Couchbase Search Service with vector search capabilities:

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

When selecting the appropriate vector index type, consider the following factors:

- **Dataset Size**: The volume of documents you plan to index.
- **Query Requirements**: Whether your searches involve pure vector queries or hybrid searches combining vectors with scalar values, text, or geospatial data.
- **Performance Needs**: The desired speed and efficiency of your search operations.

**Recommendations:**

- **Start with a Hyperscale Vector Index**: In most cases, begin by testing with a **Hyperscale Vector Index**. If performance doesn't meet your requirements, consider the other index types based on your specific use case.
- **For Hybrid Searches**: If your dataset is under 100 million documents and you need to perform hybrid searches combining vector searches with text or geospatial data, use a **Search Vector Index**.
- **For Scalar Filtering**: If you need to combine vector searches with scalar filters where scalar values are less selective (20% or less) or where scalars should exclude vectors for compliance purposes, use a **Composite Vector Index**.

#### Comparison Table

| Feature                     | Hyperscale Vector Index                                                                                                                                                                           | Composite Vector Index                                                                                                                                                                           | Search Vector Index                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Available Since**         | Couchbase Server 8.0                                                                                                                                                                              | Couchbase Server 8.0                                                                                                                                                                             | Couchbase Server 7.6                                                                                   |
| **Dataset Size**            | Tens of millions to billions of documents                                                                                                                                                         | Tens of millions to a billion                                                                                                                                                                    | Tens of millions (limited to ~100M documents)                                                          |
| **Best For**                | Pure vector searches                                                                                                                                                                              | Searches combining vector and scalar values where scalar values are less selective (≤20%); searches where scalars should exclude vectors (e.g., compliance)                                      | Hybrid searches combining vector with keywords or geospatial data                                      |
| **Use Cases**               | • Content discovery<br>• Recommendations<br>• RAG workflows<br>• Reverse image search<br>• Anomaly detection                                                                                      | • Job search<br>• Content recommendations with scalar filters<br>• Supply chain management<br>• Compliance-based filtering                                                                       | • E-commerce product search<br>• Travel recommendations<br>• Real estate search                        |
| **Scalar Handling**         | Scalars and vectors evaluated simultaneously                                                                                                                                                      | Scalar values pre-filter data before vector search                                                                                                                                               | Searches performed in parallel                                                                         |
| **Strengths**               | • High performance for pure vector searches<br>• Higher accuracy at lower quantizations<br>• Low memory footprint<br>• Lowest TCO for huge datasets<br>• Best for concurrent updates and searches | • Scalar pre-filtering reduces vector search scope<br>• Efficient when scalar values have low selectivity<br>• Can restrict searches based on scalars for compliance<br>• Based on familiar GSIs | • Combines semantic, Full-Text Search, and geospatial in single pass<br>• Uses familiar Search indexes |
| **Limitations**             | Indexing can take longer relative to other index types                                                                                                                                            | • Lower accuracy than Hyperscale at lower quantizations<br>• Scalar filtering potentially misses relevant results                                                                                | • Less efficient for purely numeric/scalar searches<br>• Limited to ~100M documents                    |
| **LlamaIndex Vector Store** | `CouchbaseQueryVectorStore`                                                                                                                                                                       | `CouchbaseQueryVectorStore`                                                                                                                                                                      | `CouchbaseSearchVectorStore`                                                                           |

> **Note on Scalar Handling:** A key difference between Hyperscale and Composite Vector indexes is how they handle scalar values in queries. Hyperscale Vector indexes compare vectors and scalar values at the same time. Composite Vector indexes always apply scalar filters first, and only perform vector searches on the results. This behavior means Composite Index searches can exclude relevant vectors from the search result. However, it’s useful for cases where you must exclude some vectors (even the nearest neighbors) based on scalar values. For example, it’s useful when meeting compliance requirements.

For more information, refer to: [Couchbase Vector Search Documentation](https://docs.couchbase.com/server/current/vector-index/use-vector-indexes.html)

## License

MIT
