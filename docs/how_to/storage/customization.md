# Customizing Storage

By default, LlamaIndex hides away the complexities and let you query your data in under 5 lines of code:
```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("Summarize the documents.")
```

Under the hood, LlamaIndex also supports a swappable **storage layer** that allows you to customize where ingested documents (i.e., `Node` objects), embedding vectors, and index metadata are stored.


![](/_static/storage/storage.png)

### Low-Level API
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
    docstore=SimpleDocumentStore(),
    vector_store=SimpleVectorStore(),
    index_store=SimpleIndexStore(),
)

# create (or load) docstore and add nodes
storage_context.docstore.add_documents(nodes)

# build index
index = GPTVectorStoreIndex(nodes, storage_context=storage_context)
```
You can customize the underlying storage with a one-line change to instantiate different document stores, index stores, and vector stores.
See [Document Stores](/how_to/storage/docstores.md), [Vector Stores](/how_to/storage/vector_stores.md), [Index Stores](/how_to/storage/index_stores.md) guides for more details.
