# Document Stores
Document stores contain ingested document chunks, which we call `Node` objects.

See the [API Reference](/reference/storage/docstore.rst) for more details.


### Simple Document Store
By default, the `SimpleDocumentStore` stores `Node` objects in-memory. 
They can be persisted to (and loaded from) disk by calling `docstore.persist()` (and `SimpleDocumentStore.from_persist_path(...)` respectively).

### BaseQueryTransformDB Document Store
We support MongoDB as an alternative document store backend that persists data as `Node` objects are ingested.
```python
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.node_parser import SimpleNodeParser

# create parser and parse document into nodes 
parser = SimpleNodeParser()
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





