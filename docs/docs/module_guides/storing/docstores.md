# Document Stores

Document stores contain ingested document chunks, which we call `Node` objects.

See the [API Reference](../../api_reference/storage/docstore/index.md) for more details.

### Simple Document Store

By default, the `SimpleDocumentStore` stores `Node` objects in-memory.
They can be persisted to (and loaded from) disk by calling `docstore.persist()` (and `SimpleDocumentStore.from_persist_path(...)` respectively).

A more complete example can be found [here](../../examples/docstore/DocstoreDemo.ipynb)

### MongoDB Document Store

We support MongoDB as an alternative document store backend that persists data as `Node` objects are ingested.

```python
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.node_parser import SentenceSplitter

# create parser and parse document into nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# create (or load) docstore and add nodes
docstore = MongoDocumentStore.from_uri(uri="<mongodb+srv://...>")
docstore.add_documents(nodes)

# create storage context
storage_context = StorageContext.from_defaults(docstore=docstore)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

Under the hood, `MongoDocumentStore` connects to a fixed MongoDB database and initializes new collections (or loads existing collections) for your nodes.

> Note: You can configure the `db_name` and `namespace` when instantiating `MongoDocumentStore`, otherwise they default to `db_name="db_docstore"` and `namespace="docstore"`.

Note that it's not necessary to call `storage_context.persist()` (or `docstore.persist()`) when using an `MongoDocumentStore`
since data is persisted by default.

You can easily reconnect to your MongoDB collection and reload the index by re-initializing a `MongoDocumentStore` with an existing `db_name` and `collection_name`.

A more complete example can be found [here](../../examples/docstore/MongoDocstoreDemo.ipynb)

### Redis Document Store

We support Redis as an alternative document store backend that persists data as `Node` objects are ingested.

```python
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter

# create parser and parse document into nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# create (or load) docstore and add nodes
docstore = RedisDocumentStore.from_host_and_port(
    host="127.0.0.1", port="6379", namespace="llama_index"
)
docstore.add_documents(nodes)

# create storage context
storage_context = StorageContext.from_defaults(docstore=docstore)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

Under the hood, `RedisDocumentStore` connects to a redis database and adds your nodes to a namespace stored under `{namespace}/docs`.

> Note: You can configure the `namespace` when instantiating `RedisDocumentStore`, otherwise it defaults `namespace="docstore"`.

You can easily reconnect to your Redis client and reload the index by re-initializing a `RedisDocumentStore` with an existing `host`, `port`, and `namespace`.

A more complete example can be found [here](../../examples/docstore/RedisDocstoreIndexStoreDemo.ipynb)

### Firestore Document Store

We support Firestore as an alternative document store backend that persists data as `Node` objects are ingested.

```python
from llama_index.storage.docstore.firestore import FirestoreDocumentStore
from llama_index.core.node_parser import SentenceSplitter

# create parser and parse document into nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# create (or load) docstore and add nodes
docstore = FirestoreDocumentStore.from_database(
    project="project-id",
    database="(default)",
)
docstore.add_documents(nodes)

# create storage context
storage_context = StorageContext.from_defaults(docstore=docstore)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

Under the hood, `FirestoreDocumentStore` connects to a firestore database in Google Cloud and adds your nodes to a namespace stored under `{namespace}/docs`.

> Note: You can configure the `namespace` when instantiating `FirestoreDocumentStore`, otherwise it defaults `namespace="docstore"`.

You can easily reconnect to your Firestore database and reload the index by re-initializing a `FirestoreDocumentStore` with an existing `project`, `database`, and `namespace`.

A more complete example can be found [here](../../examples/docstore/FirestoreDemo.ipynb)

### Couchbase Document Store

We support Couchbase as an alternative document store backend that persists data as `Node` objects are ingested.

```python
from llama_index.storage.docstore.couchbase import CouchbaseDocumentStore
from llama_index.core.node_parser import SentenceSplitter

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

# create parser and parse document into nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# create (or load) docstore and add nodes
docstore = CouchbaseDocumentStore.from_couchbase_client(
    client=cluster,
    bucket_name="llama-index",
    scope_name="_default",
    namespace="default",
)
docstore.add_documents(nodes)

# create storage context
storage_context = StorageContext.from_defaults(docstore=docstore)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

Under the hood, `CouchbaseDocumentStore` connects to a Couchbase operational database and adds your nodes to a collection named under `{namespace}_data` in the specified `{bucket_name}` and `{scope_name}`.

> Note: You can configure the `namespace`, `bucket` and `scope` when instantiating `CouchbaseIndexStore`. By default, the collection used is `docstore_data`. Apart from alphanumeric characters, `-`, `_` and `%` are only allowed as part of the collection name. The store will automatically convert other special characters to `_`.

You can easily reconnect to your Couchbase database and reload the index by re-initializing a `CouchbaseDocumentStore` with an existing `client`, `bucket_name`, `scope_name` and `namespace`.

### Tablestore Document Store

We support Tablestore as an alternative document store backend that persists data as `Node` objects are ingested.

```python
from llama_index.core import Document
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from llama_index.storage.docstore.tablestore import TablestoreDocumentStore

# create parser and parse document into nodes
parser = SentenceSplitter()
documents = [
    Document(text="I like cat.", id_="1", metadata={"key1": "value1"}),
    Document(text="Mike likes dog.", id_="2", metadata={"key2": "value2"}),
]
nodes = parser.get_nodes_from_documents(documents)

# create (or load) doc_store and add nodes
docs_tore = TablestoreDocumentStore.from_config(
    endpoint="<tablestore_end_point>",
    instance_name="<tablestore_instance_name>",
    access_key_id="<tablestore_access_key_id>",
    access_key_secret="<tablestore_access_key_secret>",
)
docs_tore.add_documents(nodes)

# create storage context
storage_context = StorageContext.from_defaults(docstore=docs_tore)

# build index
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

Under the hood, `TablestoreDocumentStore` connects to a Tablestore database and adds your nodes to a table named under `{namespace}_data`.

> Note: You can configure the `namespace` when instantiating `TablestoreDocumentStore`.

You can easily reconnect to your Tablestore database and reload the index by re-initializing a `TablestoreDocumentStore` with an existing `endpoint`, `instance_name`, `access_key_id` and `access_key_secret`.

A more complete example can be found [here](../../examples/docstore/TablestoreDocstoreDemo.ipynb)
