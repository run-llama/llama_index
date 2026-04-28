# LlamaIndex Vector Store Integration: Actian VectorAI

A LlamaIndex vector store backend powered by [Actian VectorAI DB](https://docs.vectoraidb.actian.com).

## Installation

```bash
pip install llama-index-vector-stores-actian-vectorai
```

## Requirements

- Python 3.10+
- A running Actian VectorAI DB instance (default endpoint: `localhost:6574`)

## Usage

### Basic (synchronous, context manager)

```python
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery

with ActianVectorAIVectorStore() as vector_store:
    # Add nodes
    vector_store.add(nodes)

    # Query
    query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=5)
    result = vector_store.query(query)

    # Delete by source document ID
    vector_store.delete("doc_01")

    # Delete the entire collection
    vector_store.clear()
```

### Async (context manager)

```python
async with ActianVectorAIVectorStore() as vector_store:
    await vector_store.async_add(nodes)

    query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=5)
    result = await vector_store.aquery(query)

    await vector_store.adelete("doc_01")
    await vector_store.aclear()
```

### Manual connection management

```python
vector_store = ActianVectorAIVectorStore()

vector_store.connect()
vector_store.add(nodes)
result = vector_store.query(query)
vector_store.close()
```

Async equivalent:

```python
await vector_store.aconnect()
await vector_store.async_add(nodes)
result = await vector_store.aquery(query)
await vector_store.aclose()
```

### External client

Pass a pre-configured `VectorAIClient` (or `AsyncVectorAIClient`) when you need to
share a connection or supply custom client configuration:

```python
from actian_vectorai import VectorAIClient, AsyncVectorAIClient

# Sync
with VectorAIClient("localhost:6574") as client:
    vector_store = ActianVectorAIVectorStore(client=client)
    vector_store.add(nodes)
    result = vector_store.query(query)

# Async
async with AsyncVectorAIClient("localhost:6574") as async_client:
    vector_store = ActianVectorAIVectorStore(async_client=async_client)
    await vector_store.async_add(nodes)
    result = await vector_store.aquery(query)
```

## Constructor Parameters

| Parameter                   | Type                          | Default                      | Description                                                                                                                                               |
| --------------------------- | ----------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `url`                       | `str`                         | `"localhost:6574"`           | Actian VectorAI DB endpoint (`host:port`). Ignored when explicit clients are provided.                                                                    |
| `collection_name`           | `str`                         | `"llama_index_collection"`   | Collection to use for storing vectors and metadata.                                                                                                       |
| `dense_vector_name`         | `str`                         | `"llama_index_dense_vector"` | Name of the dense vector field inside the collection.                                                                                                     |
| `dense_vector_params`       | `VectorParams \| None`        | `None`                       | Vector configuration (size, distance metric). Inferred from the first inserted embedding if omitted (defaults to cosine distance).                        |
| `stores_text`               | `bool`                        | `False`                      | Store node text in the point payload in addition to metadata.                                                                                             |
| `clear_existing_collection` | `bool`                        | `False`                      | Delete any existing collection with the same name before the first operation.                                                                             |
| `client_kwargs`             | `dict \| None`                | `None`                       | Extra keyword arguments forwarded to internally created sync/async clients.                                                                               |
| `collection_kwargs`         | `dict \| None`                | `None`                       | Extra keyword arguments passed to collection creation. Do not include `vectors_config`; it is derived from `dense_vector_name` and `dense_vector_params`. |
| `client`                    | `VectorAIClient \| None`      | `None`                       | Pre-configured synchronous client. When provided, `url` and `client_kwargs` are ignored.                                                                  |
| `async_client`              | `AsyncVectorAIClient \| None` | `None`                       | Pre-configured asynchronous client. Must be a different instance from the internal async client of a provided `client`.                                   |

## Custom Vector Configuration

```python
from actian_vectorai import VectorParams, Distance

with ActianVectorAIVectorStore(
    url="localhost:6574",
    collection_name="my_collection",
    dense_vector_name="dense_vector",
    dense_vector_params=VectorParams(size=1536, distance=Distance.Cosine),
) as vector_store:
    vector_store.add(nodes)
```

## Metadata Filtering

Metadata filters can be used with `query`, `delete_nodes`, and `adelete_nodes`.

### Supported filter operators

| Operator      | Notes                                                          |
| ------------- | -------------------------------------------------------------- |
| `EQ`          | Exact match (string or numeric)                                |
| `NE`          | Not equal                                                      |
| `GT` / `LT`   | Numeric greater/less than                                      |
| `GTE` / `LTE` | Numeric greater/less than or equal                             |
| `IN`          | Match any value in a list (or comma-separated string)          |
| `NIN`         | Match none of the values in a list (or comma-separated string) |
| `TEXT_MATCH`  | Case-sensitive substring/token match                           |
| `IS_EMPTY`    | Field is absent or null                                        |

Unsupported operators (`ANY`, `ALL`, `TEXT_MATCH_INSENSITIVE`, `CONTAINS`) raise `NotImplementedError`.

### Filter conditions

`AND`, `OR`, and `NOT` conditions are supported via `FilterCondition`.

```python
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

# AND condition
query = VectorStoreQuery(
    query_embedding=embedding,
    similarity_top_k=5,
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", operator=FilterOperator.EQ, value="ai"
            ),
            MetadataFilter(
                key="score", operator=FilterOperator.GTE, value=0.7
            ),
        ],
        condition=FilterCondition.AND,
    ),
)

# Delete nodes matching a filter
vector_store.delete_nodes(
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                operator=FilterOperator.IN,
                value=["ai", "energy"],
            ),
        ]
    )
)
```

## Limitations

- Only `VectorStoreQueryMode.DEFAULT` (dense vector search) is supported.

## Running Tests

Tests require a running Actian VectorAI DB instance. Set `VECTORAI_SERVER_URL` to override the default endpoint:

```bash
export VECTORAI_SERVER_URL="localhost:6574"
pytest
```
