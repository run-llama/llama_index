# Azure Cosmos DB for NoSQL Vector Store

This integration enables [Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/)
as a vector store in LlamaIndex, with support for vector search, full text search, and hybrid (RRF) search.

## Installation

```sh
pip install llama-index-vector-stores-azurecosmosnosql
```

## Quick Start

### Create the client

```python
from azure.cosmos import CosmosClient, PartitionKey

URI = "https://<account>.documents.azure.com:443/"
KEY = "<your-key>"
client = CosmosClient(URI, credential=KEY)
```

Alternatively, use the built-in factory methods:

```python
from llama_index.vector_stores.azurecosmosnosql import AzureCosmosDBNoSqlVectorSearch

# From host + key
store = AzureCosmosDBNoSqlVectorSearch.from_host_and_key(
    host=URI, key=KEY, ...
)

# From connection string
store = AzureCosmosDBNoSqlVectorSearch.from_connection_string(
    connection_string="AccountEndpoint=...;AccountKey=...;", ...
)

# From managed identity
store = AzureCosmosDBNoSqlVectorSearch.from_uri_and_managed_identity(
    cosmos_uri=URI, ...
)
```

### Define policies

```python
indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 3072,
        }
    ]
}
```

### Create the vector store

```python
from azure.cosmos import PartitionKey

store = AzureCosmosDBNoSqlVectorSearch(
    cosmos_client=client,
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    cosmos_database_properties={},
    database_name="myDB",
    container_name="myContainer",
)
```

### Build an index

```python
from llama_index.core import VectorStoreIndex, StorageContext

storage_context = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

---

## Search Types

All search types are available via the `search_type` kwarg on `store.query()`.

### Vector Search

Standard nearest-neighbour search ranked by `VectorDistance`.

```python
from llama_index.core.vector_stores.types import VectorStoreQuery

result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    search_type="vector",
)
```

### Vector Search with Score Threshold

Returns only nodes whose cosine similarity exceeds `threshold`.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=10),
    search_type="vector_score_threshold",
    threshold=0.8,
)
```

### Full Text Search

Filters nodes using a `FullTextContains` predicate. Requires `full_text_search_enabled=True`
and a container created with a `full_text_policy` and `fullTextIndexes`.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=10),
    search_type="full_text_search",
    where="FullTextContains(c.text, 'neural network')",
)
```

### Full Text Ranking

Ranks nodes by `FullTextScore` using `ORDER BY RANK`. Multiple `full_text_rank_filter`
entries are fused with `ORDER BY RANK RRF(...)`.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    search_type="full_text_ranking",
    full_text_rank_filter=[
        {"search_field": "text", "search_text": "neural network"},
    ],
)
```

### Hybrid Search (RRF)

Fuses `FullTextScore` and `VectorDistance` rankings using Reciprocal Rank Fusion.
Uses `OFFSET 0 LIMIT k` instead of `TOP k` as required by CosmosDB.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    search_type="hybrid",
    full_text_rank_filter=[
        {"search_field": "text", "search_text": "neural network"},
    ],
)
```

### Hybrid Search with Score Threshold

Same as hybrid but with client-side score filtering.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    search_type="hybrid_score_threshold",
    full_text_rank_filter=[
        {"search_field": "text", "search_text": "neural network"},
    ],
    threshold=0.5,
)
```

### Weighted Hybrid Search

Runs the same RRF query as `hybrid` but applies **client-side per-component
weights** when re-ranking the returned results. This lets you tune the relative
influence of each ranking component without changing the CosmosDB query.

`weights` is a list of floats whose length equals the number of
`full_text_rank_filter` entries **plus one** (the final entry is the weight for
the vector component). The values do not need to sum to 1.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    search_type="weighted_hybrid_search",
    full_text_rank_filter=[
        {"search_field": "text", "search_text": "neural network"},
    ],
    # weights=[text_weight, vector_weight]
    # 30 % text relevance, 70 % vector similarity
    weights=[0.3, 0.7],
)
```

With two full-text components and one vector component:

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    search_type="weighted_hybrid_search",
    full_text_rank_filter=[
        {"search_field": "title",   "search_text": "neural network"},
        {"search_field": "summary", "search_text": "deep learning"},
    ],
    # weights=[title_weight, summary_weight, vector_weight]
    weights=[0.2, 0.3, 0.5],
)
```

> **Note:** The CosmosDB NoSQL API does not expose a `weight=` parameter inside
> `ORDER BY RANK RRF(...)`. The weighted re-ranking is therefore performed
> client-side using position-based RRF scores after the server returns results.

---

## Query Options

### WHERE filter

Apply a CosmosDB SQL predicate to restrict results.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    search_type="vector",
    where="c.metadata.author = 'Stephen King'",
)
```

### Pagination (OFFSET / LIMIT)

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=10),
    search_type="vector",
    offset_limit="OFFSET 10 LIMIT 5",
)
```

### Projection mapping

Return only specific fields, surfaced as node metadata keys.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=3),
    search_type="vector",
    projection_mapping={"id": "id", "text": "body"},
)
```

### Return vector embeddings

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=3),
    search_type="vector",
    return_with_vectors=True,
)
```

### Backward-compatible `pre_filter`

The legacy `pre_filter` dict is still supported for existing callers.

```python
result = store.query(
    VectorStoreQuery(query_embedding=[...], similarity_top_k=5),
    pre_filter={
        "where_clause": "WHERE c.metadata.year = 2024",
        "limit_offset_clause": "OFFSET 0 LIMIT 5",
    },
)
```

---

## Full Text Search Setup

To enable full text search or hybrid search, create the container with a
`full_text_policy` and include `fullTextIndexes` in the indexing policy:

```python
full_text_indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
    "fullTextIndexes": [{"path": "/text"}],
}

full_text_policy = {
    "defaultLanguage": "en-US",
    "fullTextPaths": [{"path": "/text", "language": "en-US"}],
}

store = AzureCosmosDBNoSqlVectorSearch(
    cosmos_client=client,
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=full_text_indexing_policy,
    cosmos_container_properties={
        "partition_key": PartitionKey(path="/id"),
        "full_text_policy": full_text_policy,
    },
    cosmos_database_properties={},
    full_text_search_enabled=True,
)
```
