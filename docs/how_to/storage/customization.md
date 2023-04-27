
# Customize Storage

By default, LlamaIndex hides away the complexities and let you query your data in under 5 lines of code:
```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Summarize the documents.")
```

Under the hood, LlamaIndex also supports a swappable **storage layer** that allows you to customize where ingested documents (i.e., `Node` objects), embedding vectors, and index metadata are stored.

To do this, instead of the high-level API,
```python
index = GPTVectorStoreIndex.from_documents(documents)
```
we use a lower-level API that gives more granular control:
```python
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.node_parser import SimpleNodeParser

# create parser and parse document into nodes 
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# create storage context
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="<persist_dir>"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="<persist_dir>"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="<persist_dir>"),
)

# create (or load) docstore and add nodes
storage_context.docstore.add_documents(nodes)

# build index
index = GPTVectorStoreIndex(nodes, storage_context=storage_context)
```
You can customize the underlying storage with a one-line change to instantiate different document stores, index stores, and vector stores.
See [Document Stores](/docs/how_to/storage/docstores.md), [Vector Stores](/docs/how_to/storage/vector_stores.md), [Index Stores](/docs/how_to/storage/index_stores.md) guides for more details.

## Saving Data
By default, LlamaIndex stores data in-memory, and need to be explicitly persisted if desired:
```python
storage_context.persist()
```
This will persist data to disk, under the specified `persist_dir` (or `./storage` by default).

User can also configure alternative storage backends (e.g. `MongoDB`) that persist data by default.
In this case, calling `storage_context.persist()` will do nothing.

## Loading Data
To load data, user simply needs to re-create the storage context using the same configuration (e.g. pass in the same `persist_dir` or vector store client).

```python
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="<persist_dir>"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="<persist_dir>"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="<persist_dir>"),
)
```

To load the previously constructed indices:
```python
from llama_index import load_index_from_storage, load_indices_from_storage, load_graph_from_storage

# load a single index
index = load_index_from_storage(storage_context, index_id="<index_id>") # need to specify index_id if it's ambiguous
index = load_index_from_storage(storage_context) # don't need to specify index_id if there's only one index in storage context

# load multiple indices
indices = load_indice_from_storage(storage_context) # loads all indices
indices = load_indice_from_storage(storage_context, index_ids=<index_ids>) # loads specific indices

# load composable graph
graph = load_graph_from_storage(storage_context, root_id="<root_id>") # loads graph with the specified root_id

```



