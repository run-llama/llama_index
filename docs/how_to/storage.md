# Document Store

LlamaIndex provides a high-level interface for ingesting, indexing, and querying your external data.

By default, LlamaIndex hides away the complexities and let you query your data in under 5 lines of code:
```python
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)
response = index.query("Summarize the documents.")
```

Under the hood, LlamaIndex also supports a swappable **storage layer** that allows you to customize where ingested data (i.e., `Node` objects) are stored.

To do this, instead of the high-level API,

```python
index = GPTSimpleVectorIndex.from_documents(documents)
```
we use a lower-level API that gives more granular control:
```python
from llama_index.docstore import SimpleDocumentStore
from llama_index.node_parser import SimpleNodeParser

# create parser and parse document into nodes 
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# create document store and add nodes
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# build index
index = GPTSimpleVectorIndex(nodes, docstore=docstore)
```
You can customize the underlying storage with a one-line change to instantiate a different document store.

### Simple Document Store
By default, the `SimpleDocumentStore` stores `Node` objects in-memory. They can be persisted to (and loaded from) disk by calling `index.save_to_disk(...)` (and `Index.load_from_disk(...)` respectively).

### MongoDB Document Store
We support MongoDB as an alternative document store backend that persists data as `Node` objects are ingested.
```python
from llama_index.docstore import MongoDocumentStore
from llama_index.node_parser import SimpleNodeParser

# create parser and parse document into nodes 
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# create document store and add nodes
docstore = MongoDocumentStore.from_uri(uri="<mongodb+srv://...>")
docstore.add_documents(nodes)

# build index
index = GPTSimpleVectorIndex(nodes, docstore=docstore)
```

Under the hood, `MongoDocumentStore` connects to a fixed MongoDB database and initializes a new collection for your nodes.
> Note: You can configure the `db_name` and `collection_name` when instantiating `MongoDocumentStore`, otherwise they default to `db_name=db_docstore` and `collection_name=collection_<uuid>`.

When using `MongoDocumentStore`, calling `index.save_to_disk(...)` only saves the MongoDB connection config to disk (instead of the `Node` objects).

You can easily reconnect to your MongoDB collection and reload the index by calling `Index.load_from_disk(...)` (or by explicitly initializing a `MongoDocumentStore` with an exiting `db_name` and `collection_name`).





