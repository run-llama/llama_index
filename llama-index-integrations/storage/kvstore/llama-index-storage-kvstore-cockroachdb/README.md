# LlamaIndex Storage_Kvstore Integration: CockroachDB

A key-value store backed by [CockroachDB](https://www.cockroachlabs.com/). Used as the underlying storage for `CockroachDBDocumentStore` and `CockroachDBIndexStore`, and can also be used directly.

Uses the `cockroachdb+psycopg2` and `cockroachdb+asyncpg` SQLAlchemy dialects, so transactions retry transparently on `SERIALIZATION_FAILURE`. Values are stored in a `JSONB` column with `(collection, key)` as the primary key.

## Installation

```bash
pip install llama-index-storage-kvstore-cockroachdb
```

## Usage

```python
from llama_index.storage.kvstore.cockroachdb import CockroachDBKVStore

kvstore = CockroachDBKVStore.from_params(
    host="localhost",
    port=26257,
    database="defaultdb",
    user="root",
    password="",
    sslmode="disable",  # local insecure cluster only
    table_name="kv",
)

kvstore.put("hello", {"value": 1})
print(kvstore.get("hello"))  # {"value": 1}
```

Async variant: `aput`, `aget`, `aput_all`, `aget_all`, `adelete`.
