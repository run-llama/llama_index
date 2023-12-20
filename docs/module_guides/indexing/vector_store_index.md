# Using VectorStoreIndex

Vector Stores are a key component of retrieval-augmented generation (RAG) and so you will end up using them in nearly every application you make using LlamaIndex, either directly or indirectly.

Vector stores accept a list of [`Node` objects](/module_guides/loading/documents_and_nodes/root.md) and build an index from them

## Loading data into the index

### Basic usage

The simplest way to use a Vector Store is to load a set of documents and build an index from them using `from_documents`:

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents and build index
documents = SimpleDirectoryReader(
    "../../examples/data/paul_graham"
).load_data()
index = VectorStoreIndex.from_documents(documents)
```

```{tip}
If you are using `from_documents` on the command line, it can be convenient to pass `show_progress=True` to display a progress bar during index construction.
```

When you use `from_documents`, your Documents are split into chunks and parsed into [`Node` objects](/module_guides/loading/documents_and_nodes/root.md), lightweight abstractions over text strings that keep track of metadata and relationships.

For more on how to load documents, see [Understanding Loading](/module_guides/loading/loading.md).

By default, VectorStoreIndex stores everything in memory. See [Using Vector Stores](#using-vector-stores) below for more on how to use persistent vector stores.

```{tip}
By default, the `VectorStoreIndex` will generate and insert vectors in batches of 2048 nodes. If you are memory constrained (or have a surplus of memory), you can modify this by passing `insert_batch_size=2048` with your desired batch size.

This is especially helpful when you are inserting into a remotely hosted vector database.
```

### Using the ingestion pipeline to create nodes

If you want more control over how your documents are indexed, we recommend using the ingestion pipeline. This allows you to customize the chunking, metadata, and embedding of the nodes.

```python
from llama_index import Document
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline, IngestionCache

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
        OpenAIEmbedding(),
    ]
)

# run the pipeline
nodes = pipeline.run(documents=[Document.example()])
```

```{tip}
You can learn more about [how to use the ingestion pipeline](/module_guides/loading/ingestion_pipeline/root.md).
```

### Creating and managing nodes directly

If you want total control over your index you can [create and define nodes manually](/module_guides/loading/documents_and_nodes/usage_nodes.md) and pass them directly to the index constructor:

```python
from llama_index.schema import TextNode

node1 = TextNode(text="<text_chunk>", id_="<node_id>")
node2 = TextNode(text="<text_chunk>", id_="<node_id>")
nodes = [node1, node2]
index = VectorStoreIndex(nodes)
```

#### Handling Document Updates

When managing your index directly, you will want to deal with data sources that change over time. `Index` classes have **insertion**, **deletion**, **update**, and **refresh** operations and you can learn more about them below:

```{toctree}
---
maxdepth: 1
---
metadata_extraction.md
document_management.md
```

## Storing the vector index

LlamaIndex supports [dozens of vector stores](/module_guides/storing/vector_stores.md). You can specify which one to use by passing in a `StorageContext`, on which in turn you specify the `vector_store` argument, as in this example using Pinecone:

```python
import pinecone
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import PineconeVectorStore

# init pinecone
pinecone.init(api_key="<api_key>", environment="<environment>")
pinecone.create_index(
    "quickstart", dimension=1536, metric="euclidean", pod_type="p1"
)

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pinecone.Index("quickstart"))
)

# Load documents and build index
documents = SimpleDirectoryReader(
    "../../examples/data/paul_graham"
).load_data()
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

For more examples of how to use VectorStoreIndex, see our [vector store index usage examples notebook](./vector_store_guide.ipynb).

For examples of how to use VectorStoreIndex with specific vector stores, check out our [vector stores](/module_guides/storing/vector_stores.md) section under Loading.

```{toctree}
---
maxdepth: 1
hidden: True
---
vector_store_guide.ipynb
```
