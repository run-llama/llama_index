# Using Vector Stores

GPT Index offers multiple integration points with vector stores / vector databases:

1. GPT Index can load data from vector stores, similar to any other data connector. This data can then be used within GPT Index data structures.
2. GPT Index can use a vector store itself as an index. Like any other index, this index can store documents and be used to answer queries.

## Loading Data from Vector Stores using Data Connector

GPT Index supports loading data from the following sources. See [Data Connectors](data_connectors.md) for more details and API documentation.

- Chroma (`ChromaReader`) [Installation](https://docs.trychroma.com/getting-started)
- Qdrant (`QdrantReader`) [Installation](https://qdrant.tech/documentation/install/) [Python Client](https://qdrant.tech/documentation/install/#python-client)
- Weaviate (`WeaviateReader`). [Installation](https://weaviate.io/developers/weaviate/current/getting-started/installation.html). [Python Client](https://weaviate.io/developers/weaviate/current/client-libraries/python.html).
- Pinecone (`PineconeReader`). [Installation/Quickstart](https://docs.pinecone.io/docs/quickstart).
- Faiss (`FaissReader`). [Installation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

Chroma stores both documents and vectors. This is an example of how to use Chroma:

```python

from gpt_index.readers.chroma import ChromaReader
from gpt_index.indices import GPTListIndex

# The chroma reader loads data from a persisted Chroma collection.
# This requires a collection name and a persist directory.
reader = ChromaReader(
    collection_name="chroma_collection",
    persist_directory="examples/data_connectors/chroma_collection"
)

query_vector=[n1, n2, n3, ...]

documents = reader.load_data(collection_name="demo", query_vector=query_vector, limit=5)
index = GPTListIndex(documents)

response = index.query("<query_text>")
display(Markdown(f"<b>{response}</b>"))
```

Qdrant also stores both documents and vectors. This is an example of how to use Qdrant:

![](/_static/vector_stores/qdrant_reader.png)

NOTE: Since Weaviate can store a hybrid of document and vector objects, the user may either choose to explicitly specify `class_name` and `properties` in order to query documents, or they may choose to specify a raw GraphQL query. See below for usage.

![](/_static/vector_stores/weaviate_reader_0.png)
![](/_static/vector_stores/weaviate_reader_1.png)

NOTE: Both Pinecone and Faiss data loaders assume that the respective data sources only store vectors; text content is stored elsewhere. Therefore, both data loaders require that the user specifies an `id_to_text_map` in the load_data call.

For instance, this is an example usage of the Pinecone data loader `PineconeReader`:

![](/_static/vector_stores/pinecone_reader.png)

[Example notebooks can be found here](https://github.com/jerryjliu/gpt_index/tree/main/examples/data_connectors).

(vector-store-index)=

## Using a Vector Store as an Index

GPT Index also supports using a vector store itself as an index.
These are found in the following classes:

- `GPTSimpleVectorIndex`
- `GPTFaissIndex`
- `GPTWeaviateIndex`
- `GPTPineconeIndex`
- `GPTQdrantIndex`
- `GPTChromaIndex`

Similar to any other index within GPT Index (tree, keyword table, list), this index can be constructed upon any collection
of documents. We use the vector store within the index to store embeddings for the input text chunks.

Once constructed, the index can be used for querying.

**Faiss Index Construction/Querying**
![](/_static/vector_stores/faiss_index_0.png)
![](/_static/vector_stores/faiss_index_1.png)

**Simple Index Construction/Querying**
![](/_static/vector_stores/simple_index_0.png)

**Weaviate Index Construction/Querying**
![](/_static/vector_stores/weaviate_index_0.png)

**Pinecone Index Construction/Querying**
![](/_static/vector_stores/pinecone_index_0.png)

**Qdrant Index Construction/Querying**
![](/_static/vector_stores/qdrant_index_0.png)

**Chroma Index Construction/Querying**

```python

import chromadb
from gpt_index import GPTChromaIndex, SimpleDirectoryReader

# By default, Chroma will operate purely in-memory.
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("quickstart")

# load documents
documents = SimpleDirectoryReader('../../examples/paul_graham_essay/data').load_data()

# N.B: OPENAI_API_KEY must be set as an environment variable.
index = GPTChromaIndex(documents, chroma_collection=chroma_collection)

response = index.query("What did the author do growing up?", chroma_collection=chroma_collection)
display(Markdown(f"<b>{response}</b>"))

```

[Example notebooks can be found here](https://github.com/jerryjliu/gpt_index/tree/main/examples/vector_indices).
