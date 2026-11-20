# LlamaIndex Cassandra Document Store Integration

This package provides a Cassandra-backed document store implementation for LlamaIndex.

## Installation

```bash
pip install llama-index-storage-docstore-cassandra
```

## Usage

```python
from llama_index.storage.docstore.cassandra import CassandraDocumentStore

# Using connection parameters
docstore = CassandraDocumentStore.from_params(
    contact_points=["127.0.0.1"],
    port=9042,
    keyspace="my_keyspace",
    table="my_kv_table"
)

# Or using an existing KVStore
from llama_index.storage.kvstore.cassandra import CassandraKVStore
kvstore = CassandraKVStore.from_params(contact_points=["127.0.0.1"])
docstore = CassandraDocumentStore.from_kvstore(kvstore)

# Use with LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, docstore=docstore)
```

## Requirements

- Python >= 3.10
- Cassandra cluster (Apache Cassandra or Astra DB)
- cassandra-driver >= 3.25.0