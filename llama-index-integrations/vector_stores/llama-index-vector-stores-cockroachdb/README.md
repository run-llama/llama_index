# LlamaIndex Vector_Stores Integration: CockroachDB

This integration adds [CockroachDB](https://www.cockroachlabs.com/) as a vector store for LlamaIndex, backed by CRDB's native `VECTOR(n)` column type and the [C-SPANN](https://www.cockroachlabs.com/docs/stable/vector.html) distributed approximate nearest neighbor index.

`llama-index-vector-stores-postgres` depends on the pgvector extension, which is not available in CockroachDB. CRDB ships its own vector primitives instead: a native `VECTOR(n)` type, a `CREATE VECTOR INDEX ... WITH (min_partition_size, max_partition_size)` DDL, and a `vector_search_beam_size` session var for runtime recall/latency trade-offs. This package targets those primitives directly.

## Installation

```bash
pip install llama-index-vector-stores-cockroachdb
```

Requires CockroachDB v25.2 or later. Enable the vector feature once at the cluster level (the store will attempt this on first init if the user has permission):

```sql
SET CLUSTER SETTING feature.vector_index.enabled = true;
```

## Usage

```python
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore

store = CockroachDBVectorStore.from_params(
    host="localhost",
    port=26257,
    user="root",
    password="",
    database="defaultdb",
    table_name="my_index",
    embed_dim=1536,
    distance_metric="cosine",
    cspann_kwargs={"min_partition_size": 16, "max_partition_size": 128},
    sslmode="disable",
)

index = VectorStoreIndex.from_documents(
    [Document(text="...")],
    storage_context=StorageContext.from_defaults(vector_store=store),
)
print(index.as_query_engine().query("..."))
```

## Tuning C-SPANN

Two levers:

1. Build-time: `cspann_kwargs={"min_partition_size": ..., "max_partition_size": ...}` on `from_params()`.
2. Query-time: `vector_search_beam_size=N` per `query()` call. Higher beam means better recall, slightly more latency. Issued as `SET LOCAL vector_search_beam_size` per session.

## Supported query modes

| Mode | Supported | Notes |
| --- | --- | --- |
| `DEFAULT` | yes | ANN through C-SPANN |
| `MMR` | yes | Client-side rerank with `mmr_threshold`, `mmr_prefetch_factor`, `mmr_prefetch_k` |
| `HYBRID` / `SPARSE` / `TEXT_SEARCH` | no | CRDB has no `tsvector` yet; raises `NotImplementedError` |

## Distance metrics

`cosine`, `l2`, `inner_product`. Picks the matching opclass (`vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops`) at index creation time.

## Metadata filters

All standard LlamaIndex operators: `EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `IN`, `NIN`, `CONTAINS`, `TEXT_MATCH`, `TEXT_MATCH_INSENSITIVE`, `IS_EMPTY`, nestable via `MetadataFilters` with AND/OR/NOT. For frequently filtered keys, declare `indexed_metadata_keys={("category", "text"), ("year", "int")}` on the store to get JSONB-extracted BTREE indices.
