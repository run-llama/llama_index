# LlamaIndex Storage_Docstore Integration: CockroachDB

A document (node) store backed by [CockroachDB](https://www.cockroachlabs.com/). Thin wrapper over `CockroachDBKVStore`.

## Installation

```bash
pip install llama-index-storage-docstore-cockroachdb
```

This pulls in `llama-index-storage-kvstore-cockroachdb` as a transitive dependency.

## Usage

```python
from llama_index.storage.docstore.cockroachdb import CockroachDBDocumentStore

doc_store = CockroachDBDocumentStore.from_params(
    host="localhost",
    port=26257,
    database="defaultdb",
    user="root",
    password="",
    sslmode="disable",  # local insecure cluster only
    table_name="docs",
)
```
