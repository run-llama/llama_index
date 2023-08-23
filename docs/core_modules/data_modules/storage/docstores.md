# Document Stores
Document stores contain ingested document chunks, which we call `Node` objects.

See the [API Reference](/api_reference/storage/docstore.rst) for more details.


### Simple Document Store
By default, the `SimpleDocumentStore` stores `Node` objects in-memory. 
They can be persisted to (and loaded from) disk by calling `docstore.persist()` (and `SimpleDocumentStore.from_persist_path(...)` respectively).

A more complete example can be found [here](../../examples/docstore/DocstoreDemo.ipynb)

### MongoDB Document Store
We support MongoDB as an alternative document store backend that persists data as `Node` objects are ingested.
```python
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.node_parser import SimpleNodeParser

# create parser and parse document into nodes 
parser = SimpleNodeParser.from_defaults()
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
from llama_index.storage.docstore import RedisDocumentStore
from llama_index.node_parser import SimpleNodeParser

# create parser and parse document into nodes 
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# create (or load) docstore and add nodes
docstore = RedisDocumentStore.from_host_and_port(
  host="127.0.0.1", 
  port="6379", 
  namespace='llama_index'
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
from llama_index.storage.docstore import FirestoreDocumentStore
from llama_index.node_parser import SimpleNodeParser

# create parser and parse document into nodes
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# create (or load) docstore and add nodes
docstore = FirestoreDocumentStore.from_dataabse(
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
