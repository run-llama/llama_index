# Using Vector Stores

LlamaIndex offers multiple integration points with vector stores / vector databases:

1. LlamaIndex can load data from vector stores, similar to any other data connector. This data can then be used within LlamaIndex data structures.
2. LlamaIndex can use a vector store itself as an index. Like any other index, this index can store documents and be used to answer queries.

## Loading Data from Vector Stores using Data Connector

LlamaIndex supports loading data from the following sources. See [Data Connectors](/how_to/data_connectors.md) for more details and API documentation.

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
index = GPTListIndex.from_documents(documents)

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

LlamaIndex also supports using a vector store itself as an index.
These are found in the following classes:
- `GPTSimpleVectorIndex`
- `GPTFaissIndex`
- `GPTWeaviateIndex`
- `GPTPineconeIndex`
- `GPTQdrantIndex`
- `GPTChromaIndex`


An API reference of each vector index is [found here](/reference/indices/vector_store.rst).

Similar to any other index within LlamaIndex (tree, keyword table, list), this index can be constructed upon any collection
of documents. We use the vector store within the index to store embeddings for the input text chunks.

Once constructed, the index can be used for querying.

**Simple Index Construction/Querying**
```python
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# Load documents, build the GPTSimpleVectorIndex
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)

# Query index
response = index.query("What did the author do growing up?")

```

**Faiss Index Construction/Querying**
```python
from gpt_index import GPTFaissIndex, SimpleDirectoryReader
import faiss

# Creating a faiss index
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# Load documents, build the GPTFaissIndex
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTFaissIndex.from_documents(documents, faiss_index=faiss_index)

# Query index
response = index.query("What did the author do growing up?")

```

**Weaviate Index Construction/Querying**
```python
from gpt_index import GPTWeaviateIndex, SimpleDirectoryReader
import weaviate

# Creating a Weaviate vector store
resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    "https://<cluster-id>.semi.network/", auth_client_secret=resource_owner_config
)

# Load documents, build the GPTWeaviateIndex
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTWeaviateIndex.from_documents(documents, weaviate_client=client)

# Query index
response = index.query("What did the author do growing up?")

```

**Pinecone Index Construction/Querying**
```python
from gpt_index import GPTPineconeIndex, SimpleDirectoryReader
import pinecone

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


# Load documents, build the GPTPineconeIndex
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTPineconeIndex.from_documents(
    documents, pinecone_index=index, metadata_filters=metadata_filters
)

# Query index
response = index.query("What did the author do growing up?")
```

**Qdrant Index Construction/Querying**
```python
import qdrant_client
from gpt_index import GPTQdrantIndex, SimpleDirectoryReader

# Creating a Qdrant vector store
client = qdrant_client.QdrantClient(
    host="<qdrant-host>",
    api_key="<qdrant-api-key>",
    https=True
)
collection_name = "paul_graham"

# Load documents, build the GPTQdrantIndex
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTQdrantIndex.from_documents(documents, collection_name=collection_name, client=client)

# Query index
response = index.query("What did the author do growing up?")
```

**Chroma Index Construction/Querying**

```python
import chromadb
from gpt_index import GPTChromaIndex, SimpleDirectoryReader

# Creating a Chroma vector store
# By default, Chroma will operate purely in-memory.
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("quickstart")

# Load documents, build the GPTChromaIndex
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTChromaIndex.from_documents(documents, chroma_collection=chroma_collection)

# Query index
response = index.query("What did the author do growing up?")

```

[Example notebooks can be found here](https://github.com/jerryjliu/gpt_index/tree/main/examples/vector_indices).
