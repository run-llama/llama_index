# LlamaIndex Storage_Index_Store Integration: CockroachDB

An index store backed by [CockroachDB](https://www.cockroachlabs.com/). Thin wrapper over `CockroachDBKVStore`.

## Installation

```bash
pip install llama-index-storage-index-store-cockroachdb
```

This pulls in `llama-index-storage-kvstore-cockroachdb` as a transitive dependency.

## Usage

```python
from llama_index.storage.index_store.cockroachdb import CockroachDBIndexStore

index_store = CockroachDBIndexStore.from_params(
    host="localhost",
    port=26257,
    database="defaultdb",
    user="root",
    password="",
    sslmode="disable",  # local insecure cluster only
    table_name="idx",
)
```
