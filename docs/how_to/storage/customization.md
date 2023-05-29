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

# create storage context using default stores
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore(),
    vector_store=SimpleVectorStore(),
    index_store=SimpleIndexStore(),
)

# create (or load) docstore and add nodes
storage_context.docstore.add_documents(nodes)

# build index
index = GPTVectorStoreIndex(nodes, storage_context=storage_context)

# save index
index.storage_context.persist(persist_dir="<persist_dir>")

# can also set index_id to save multiple indexes to the same folder
index.set_index_id = "<index_id>"
index.storage_context.persist(persist_dir="<persist_dir>")

# to load index later, make sure you setup the storage context
# this will loaded the persisted stores from persist_dir
storage_context = StorageContext.from_defaults(
    persist_dir="<persist_dir>"
)

# then load the index object
from llama_index import load_index_from_storage
loaded_index = load_index_from_storage(storage_context)

# if loading an index from a persist_dir containing multiple indexes
loaded_index = load_index_from_storage(storage_context, index_id="<index_id>")

# if loading multiple indexes from a persist dir
loaded_indicies = load_index_from_storage(storage_context, index_ids=["<index_id>", ...])
```

You can customize the underlying storage with a one-line change to instantiate different document stores, index stores, and vector stores.
See [Document Stores](/how_to/storage/docstores.md), [Vector Stores](/how_to/storage/vector_stores.md), [Index Stores](/how_to/storage/index_stores.md) guides for more details.

For saving and loading a graph/composable index, see the [full guide here](/how_to/index_structs/composability.md).

### Vector Store Integrations and Storage

Most of our vector store integrations store the entire index (vectors + text) in the vector store itself. This comes with the major benefit of not having to exlicitly persist the index as shown above, since the vector store is already hosted and persisting the data in our index.

The vector stores that support this practice are:

- ChatGPTRetrievalPluginClient
- ChromaVectorStore
- LanceDBVectorStore
- MetalVectorStore
- MilvusVectorStore
- MyScaleVectorStore
- OpensearchVectorStore
- PineconeVectorStore
- QdrantVectorStore
- RedisVectorStore
- WeaviateVectorStore

A small example using Pinecone is below:

```python
import pinecone
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore

# Creating a Pinecone index
api_key = "api_key"
pinecone.init(api_key=api_key, environment="us-west1-gcp")
pinecone.create_index(
    "quickstart",
    dimension=1536,
    metric="euclidean",
    pod_type="p1"
)
index = pinecone.Index("quickstart")

# can define filters specific to this vector index (so you can
# reuse pinecone indexes)
metadata_filters = {"title": "paul_graham_essay"}

# construct vector store
vector_store = PineconeVectorStore(
    pinecone_index=index,
    metadata_filters=metadata_filters
)

# create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load documents
documents = SimpleDirectoryReader("./data").load_data()

# create index, which will insert documents/vectors to pinecone
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

# re-build/load the index by conntect to the same vector store
loaded_index = GPTVectorStoreIndex([], storage_context=storage_context)
```
