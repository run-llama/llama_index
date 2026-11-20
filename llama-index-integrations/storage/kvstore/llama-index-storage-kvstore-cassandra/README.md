# LlamaIndex Cassandra KVStore Integration

This package provides a Cassandra-backed key-value store implementation for LlamaIndex.

## Installation

```bash
pip install llama-index-storage-kvstore-cassandra
```

## Usage

```python
from llama_index.storage.kvstore.cassandra import CassandraKVStore

# Using connection parameters
kvstore = CassandraKVStore.from_params(
    contact_points=["127.0.0.1"],
    port=9042,
    keyspace="my_keyspace",
    table="my_kv_table"
)

# Or using an existing session
from cassandra.cluster import Cluster
cluster = Cluster(["127.0.0.1"])
session = cluster.connect()
kvstore = CassandraKVStore(session=session, keyspace="my_keyspace")

# Basic operations
kvstore.put("my_key", {"data": "value"}, collection="my_collection")
value = kvstore.get("my_key", collection="my_collection")
all_values = kvstore.get_all(collection="my_collection")
kvstore.delete("my_key", collection="my_collection")
```

## With Document Store

```python
from llama_index.storage.docstore.cassandra import CassandraDocumentStore
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