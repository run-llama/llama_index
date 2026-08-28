---
title: Index Stores
---

Index stores contains lightweight index metadata (i.e. additional state information created when building an index).

See the [API Reference](/python/framework-api-reference/storage/index_store) for more details.

### Simple Index Store

By default, LlamaIndex uses a simple index store backed by an in-memory key-value store.
They can be persisted to (and loaded from) disk by calling `index_store.persist()` (and `SimpleIndexStore.from_persist_path(...)` respectively).

### MongoDB Index Store

You can also use MongoDB as the storage backend for the index store.
Before you begin, install the integration package by running the following command:

```bash
pip install llama-index-storage-index-store-mongodb
```

Then, use the following code to create a `MongoIndexStore`, connect to
your MongoDB deployment, and build or load a `VectorStoreIndex`:

```python
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

# create (or load) index store
index_store = MongoIndexStore.from_uri(uri="<mongodb+srv://...>")

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index from your parsed nodes
index = VectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load an existing index
index = load_index_from_storage(storage_context)
```

`MongoIndexStore` connects to a MongoDB database and initializes new collections
(or loads existing collections) for your index metadata.

You can use either a self-managed MongoDB deployment or a MongoDB Atlas cluster.
To learn how to retrieve a connection string, see
[Connection Strings](https://www.mongodb.com/docs/manual/reference/connection-string/)
in the MongoDB documentation.

> Note: You can configure the `db_name`, `namespace`, and `collection_suffix` when instantiating `MongoIndexStore`. Otherwise, they default to `db_name="db_docstore"`, `namespace="index_store"`, and `collection_suffix="/data"`, and `MongoIndexStore` writes to the collection `index_store/data` in the `db_docstore` database

You don't need to call `storage_context.persist()` (or `index_store.persist()`) when using an `MongoIndexStore`
since data is persisted by default.

To reconnect to your collections and reload the index, re-initialize a `MongoIndexStore` with the same `db_name`, `namespace`, and `collection_suffix` values.

For a more complete example, see the [MongoDB docstore demo](/python/examples/docstore/mongodocstoredemo).

### Redis Index Store

We support Redis as an alternative document store backend that persists data as `Node` objects are ingested.

```python
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.core import VectorStoreIndex

# create (or load) docstore and add nodes
index_store = RedisIndexStore.from_host_and_port(
    host="127.0.0.1", port="6379", namespace="llama_index"
)

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load index
from llama_index.core import load_index_from_storage

index = load_index_from_storage(storage_context)
```

Under the hood, `RedisIndexStore` connects to a redis database and adds your nodes to a namespace stored under `{namespace}/index`.

> Note: You can configure the `namespace` when instantiating `RedisIndexStore`, otherwise it defaults `namespace="index_store"`.

You can easily reconnect to your Redis client and reload the index by re-initializing a `RedisIndexStore` with an existing `host`, `port`, and `namespace`.

A more complete example can be found [here](/python/examples/docstore/redisdocstoreindexstoredemo)

### Couchbase Index Store

Couchbase can be used as the storage backend for the index store.

```python
from llama_index.storage.index_store.couchbase import CouchbaseIndexStore
from llama_index.core import VectorStoreIndex

from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from datetime import timedelta

# create couchbase client
auth = PasswordAuthenticator("DB_USERNAME", "DB_PASSWORD")
options = ClusterOptions(authenticator=auth)

cluster = Cluster("couchbase://localhost", options)

# Wait until the cluster is ready for use.
cluster.wait_until_ready(timedelta(seconds=5))

# create (or load) docstore and add nodes
index_store = CouchbaseIndexStore.from_couchbase_client(
    client=cluster,
    bucket_name="llama-index",
    scope_name="_default",
    namespace="default",
)

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load index
from llama_index.core import load_index_from_storage

index = load_index_from_storage(storage_context)
```

Under the hood, `CouchbaseIndexStore` connects to a Couchbase operational database and adds your nodes to a collection named `{namespace}_index` in the specified `{bucket_name}` and `{scope_name}`.

> Note: You can configure the `namespace`, `bucket` and `scope` when instantiating `CouchbaseIndexStore`. By default, the collection used is `index_store_data`. Apart from alphanumeric characters, `-`, `_` and `%` are only allowed as part of the collection name. The store will automatically convert other special characters to `_`.

You can easily reconnect to your Couchbase client and reload the index by re-initializing a `CouchbaseIndexStore` with an existing `client`, `bucket_name`, `scope_name` and `namespace`.

### Tablestore Index Store

Similarly to document stores, we can also use `Tablestore` as the storage backend of the index store.

```python
from llama_index.storage.index_store.tablestore import TablestoreIndexStore
from llama_index.core import StorageContext, VectorStoreIndex

# create (or load) index store
index_store = TablestoreIndexStore.from_config(
    endpoint="<tablestore_end_point>",
    instance_name="<tablestore_instance_name>",
    access_key_id="<tablestore_access_key_id>",
    access_key_secret="<tablestore_access_key_secret>",
)

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load index
from llama_index.core import load_index_from_storage

index = load_index_from_storage(storage_context)
```

Under the hood, `TablestoreIndexStore` connects to a Tablestore database and adds your nodes to a table named under `{namespace}_data`.

> Note: You can configure the `namespace` when instantiating `TablestoreIndexStore`.

You can easily reconnect to your Tablestore database and reload the index by re-initializing a `TablestoreIndexStore` with an existing `endpoint`, `instance_name`, `access_key_id` and `access_key_secret`.

A more complete example can be found [here](/python/examples/docstore/tablestoredocstoredemo)

### Google AlloyDB Index Store

Similarly to document stores, we can also use [`AlloyDB`](https://cloud.google.com/products/alloydb) as the storage backend of the index store.
This tutorial demonstrates the synchronous interface. All synchronous methods have corresponding asynchronous methods.

```bash
pip install llama-index
pip install llama-index-alloydb-pg
pip install llama-index-llms-vertex
```

```python
from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBIndexStore
from llama_index.core import StorageContext, VectorStoreIndex

# create an AlloyDB Engine for connection pool
engine = AlloyDBEngine.from_instance(
    project_id=PROJECT_ID,
    region=REGION,
    cluster=CLUSTER,
    instance=INSTANCE,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
)

# initialize a new table in AlloyDB
engine.init_index_store_table(
    table_name=TABLE_NAME,
)

index_store = AlloyDBIndexStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
)

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load index
from llama_index.core import load_index_from_storage

index = load_index_from_storage(storage_context)
```

> Note: You can configure the `schema_name` along with the `table_name` when initializing a new table and instantiating `AlloyDBIndexStore`. By default the `schema_name` is `public`.

Under the hood, `AlloyDBIndexStore` connects to the alloydb database in Google Cloud and adds your nodes to a table under the `schema_name`.

You can easily reconnect to your AlloyDB database and reload the index by re-initializing a `AlloyDBIndexStore` with an `AlloyDBEngine` without initializing a new table.

A more detailed guide can be found [here](/python/examples/docstore/alloydbdocstoredemo)

### Google Cloud SQL for PostgreSQL Index Store

Similarly to document stores, we can also use [`Cloud SQL for PostgreSQL`](https://cloud.google.com/sql) as the storage backend of the index store.
This tutorial demonstrates the synchronous interface. All synchronous methods have corresponding asynchronous methods.

```bash
pip install llama-index
pip install llama-index-cloud-sql-pg
```

```python
from llama_index_cloud_sql_pg import PostgresEngine, PostgresIndexStore
from llama_index.core import StorageContext, VectorStoreIndex

# create an Postgres Engine for connection pool
engine = PostgresEngine.from_instance(
    project_id=PROJECT_ID,
    region=REGION,
    instance=INSTANCE,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
)

# initialize a new table in cloud sql postgres
engine.init_index_store_table(
    table_name=TABLE_NAME,
)

index_store = PostgresIndexStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
)

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load index
from llama_index.core import load_index_from_storage

index = load_index_from_storage(storage_context)
```

> Note: You can configure the `schema_name` along with the `table_name` when initializing a new table and instantiating `PostgresIndexStore`. By default the `schema_name` is `public`.

Under the hood, `PostgresIndexStore` connects to the cloud sql postgres database in Google Cloud and adds your nodes to a table under the `schema_name`.

You can easily reconnect to your cloud sql postgres database and reload the index by re-initializing a `PostgresIndexStore` with an `PostgresEngine` without initializing a new table.

A more detailed guide can be found [here](/python/examples/docstore/cloudsqlpgdocstoredemo)
