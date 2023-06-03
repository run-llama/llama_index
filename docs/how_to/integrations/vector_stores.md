# Using Vector Stores

LlamaIndex offers multiple integration points with vector stores / vector databases:

1. LlamaIndex can use a vector store itself as an index. Like any other index, this index can store documents and be used to answer queries.
2. LlamaIndex can load data from vector stores, similar to any other data connector. This data can then be used within LlamaIndex data structures.

(vector-store-index)=

## Using a Vector Store as an Index

LlamaIndex also supports different vector stores
as the storage backend for `GPTVectorStoreIndex`.

- Chroma (`ChromaVectorStore`) [Installation](https://docs.trychroma.com/getting-started)
- DeepLake (`DeepLakeVectorStore`) [Installation](https://docs.deeplake.ai/en/latest/Installation.html)
- Qdrant (`QdrantVectorStore`) [Installation](https://qdrant.tech/documentation/install/) [Python Client](https://qdrant.tech/documentation/install/#python-client)
- Weaviate (`WeaviateVectorStore`). [Installation](https://weaviate.io/developers/weaviate/installation). [Python Client](https://weaviate.io/developers/weaviate/client-libraries/python).
- Pinecone (`PineconeVectorStore`). [Installation/Quickstart](https://docs.pinecone.io/docs/quickstart).
- Faiss (`FaissVectorStore`). [Installation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).
- Milvus (`MilvusVectorStore`). [Installation](https://milvus.io/docs)
- Zilliz (`MilvusVectorStore`). [Quickstart](https://zilliz.com/doc/quick_start)
- MyScale (`MyScaleVectorStore`). [Quickstart](https://docs.myscale.com/en/quickstart/). [Installation/Python Client](https://docs.myscale.com/en/python-client/).
- Supabase (`SupabaseVectorStore`). [Quickstart](https://supabase.github.io/vecs/api/).

A detailed API reference is [found here](/reference/indices/vector_store.rst).

Similar to any other index within LlamaIndex (tree, keyword table, list), `GPTVectorStoreIndex` can be constructed upon any collection
of documents. We use the vector store within the index to store embeddings for the input text chunks.

Once constructed, the index can be used for querying.

**Default Vector Store Index Construction/Querying**

By default, `GPTVectorStoreIndex` uses a in-memory `SimpleVectorStore`
that's initialized as part of the default storage context.

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

# Load documents and build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

```

**Custom Vector Store Index Construction/Querying**

We can query over a custom vector store as follows:

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import DeepLakeVectorStore

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store = DeepLakeVectorStore(dataset_path="<dataset_path>")
)

# Load documents and build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

Below we show more examples of how to construct various vector stores we support.

**Redis**
First, start Redis-Stack (or get url from Redis provider)

```bash
docker run --name redis-vecdb -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

Then connect and use Redis as a vector database with LlamaIndex

```python
from llama_index.vector_stores import RedisVectorStore
vector_store = RedisVectorStore(
    index_name="llm-project",
    redis_url="redis://localhost:6379",
    overwrite=True
)
```

This can be used with the `GPTVectorStoreIndex` to provide a query interface for retrieval, querying, deleting, persisting the index, and more.

**DeepLake**

```python
import os
import getpath
from llama_index.vector_stores import DeepLakeVectorStore

os.environ["OPENAI_API_KEY"] = getpath.getpath("OPENAI_API_KEY: ")
os.environ["ACTIVELOOP_TOKEN"] = getpath.getpath("ACTIVELOOP_TOKEN: ")
dataset_path = "hub://adilkhan/paul_graham_essay"

# construct vector store
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
```

**Faiss**

```python
import faiss
from llama_index.vector_stores import FaissVectorStore

# create faiss index
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# construct vector store
vector_store = FaissVectorStore(faiss_index)

...

# NOTE: since faiss index is in-memory, we need to explicitly call
#       vector_store.persist() or storage_context.persist() to save it to disk.
#       persist() takes in optional arg persist_path. If none give, will use default paths.
storage_context.persist()
```

**Weaviate**

```python
import weaviate
from llama_index.vector_stores import WeaviateVectorStore

# creating a Weaviate client
resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    "https://<cluster-id>.semi.network/", auth_client_secret=resource_owner_config
)

# construct vector store
vector_store = WeaviateVectorStore(weaviate_client=client)
```

**Pinecone**

```python
import pinecone
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
```

**Qdrant**

```python
import qdrant_client
from llama_index.vector_stores import QdrantVectorStore

# Creating a Qdrant vector store
client = qdrant_client.QdrantClient(
    host="<qdrant-host>",
    api_key="<qdrant-api-key>",
    https=True
)
collection_name = "paul_graham"

# construct vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
)
```

**Chroma**

```python
import chromadb
from llama_index.vector_stores import ChromaVectorStore

# Creating a Chroma client
# By default, Chroma will operate purely in-memory.
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("quickstart")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)
```

**Milvus**

- Milvus Index offers the ability to store both Documents and their embeddings. Documents are limited to the predefined Document attributes and does not include extra_info.

```python
import pymilvus
from llama_index.vector_stores import MilvusVectorStore

# construct vector store
vector_store = MilvusVectorStore(
    host='localhost',
    port=19530,
    overwrite='True'
)

```

**Note**: `MilvusVectorStore` depends on the `pymilvus` library.
Use `pip install pymilvus` if not already installed.
If you get stuck at building wheel for `grpcio`, check if you are using python 3.11
(there's a known issue: https://github.com/milvus-io/pymilvus/issues/1308)
and try downgrading.

**Zilliz**

- Zilliz Cloud (hosted version of Milvus) uses the Milvus Index with some extra arguments.

```python
import pymilvus
from llama_index.vector_stores import MilvusVectorStore


# construct vector store
vector_store = MilvusVectorStore(
    host='foo.vectordb.zillizcloud.com',
    port=403,
    user="db_admin",
    password="foo",
    use_secure=True,
    overwrite='True'
)
```

**Note**: `MilvusVectorStore` depends on the `pymilvus` library.
Use `pip install pymilvus` if not already installed.
If you get stuck at building wheel for `grpcio`, check if you are using python 3.11
(there's a known issue: https://github.com/milvus-io/pymilvus/issues/1308)
and try downgrading.

**MyScale**

```python
import clickhouse_connect
from llama_index.vector_stores import MyScaleVectorStore

# Creating a MyScale client
client = clickhouse_connect.get_client(
    host='YOUR_CLUSTER_HOST',
    port=8443,
    username='YOUR_USERNAME',
    password='YOUR_CLUSTER_PASSWORD'
)


# construct vector store
vector_store = MyScaleVectorStore(
    myscale_client=client
)
```

[Example notebooks can be found here](https://github.com/jerryjliu/llama_index/tree/main/docs/examples/vector_stores).

## Loading Data from Vector Stores using Data Connector

LlamaIndex supports oading data from the following sources. See [Data Connectors](/how_to/data_connectors.md) for more details and API documentation.

Chroma stores both documents and vectors. This is an example of how to use Chroma:

```python

from llama_index.readers.chroma import ChromaReader
from llama_index.indices import GPTListIndex

# The chroma reader loads data from a persisted Chroma collection.
# This requires a collection name and a persist directory.
reader = ChromaReader(
    collection_name="chroma_collection",
    persist_directory="examples/data_connectors/chroma_collection"
)

query_vector=[n1, n2, n3, ...]

documents = reader.load_data(collection_name="demo", query_vector=query_vector, limit=5)
index = GPTListIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
display(Markdown(f"<b>{response}</b>"))
```

Qdrant also stores both documents and vectors. This is an example of how to use Qdrant:

```python

from llama_index.readers.qdrant import QdrantReader

reader = QdrantReader(host="localhost")

# the query_vector is an embedding representation of your query_vector
# Example query_vector
# query_vector = [0.3, 0.3, 0.3, 0.3, ...]

query_vector = [n1, n2, n3, ...]

# NOTE: Required args are collection_name, query_vector.
# See the Python client: https;//github.com/qdrant/qdrant_client
# for more details

documents = reader.load_data(collection_name="demo", query_vector=query_vector, limit=5)

```

NOTE: Since Weaviate can store a hybrid of document and vector objects, the user may either choose to explicitly specify `class_name` and `properties` in order to query documents, or they may choose to specify a raw GraphQL query. See below for usage.

```python
# option 1: specify class_name and properties

# 1) load data using class_name and properties
documents = reader.load_data(
    class_name="<class_name>",
    properties=["property1", "property2", "..."],
    separate_documents=True
)

# 2) example GraphQL query
query = """
{
    Get {
        <class_name> {
            <property1>
            <property2>
        }
    }
}
"""

documents = reader.load_data(graphql_query=query, separate_documents=True)
```

NOTE: Both Pinecone and Faiss data loaders assume that the respective data sources only store vectors; text content is stored elsewhere. Therefore, both data loaders require that the user specifies an `id_to_text_map` in the load_data call.

For instance, this is an example usage of the Pinecone data loader `PineconeReader`:

```python

from llama_index.readers.pinecone import PineconeReader

reader = PineconeReader(api_key=api_key, environment="us-west1-gcp")

id_to_text_map = {
    "id1": "text blob 1",
    "id2": "text blob 2",
}

query_vector=[n1, n2, n3, ..]

documents = reader.load_data(
    index_name="quickstart", id_to_text_map=id_to_text_map, top_k=3, vector=query_vector, separate_documents=True
)

```

[Example notebooks can be found here](https://github.com/jerryjliu/llama_index/tree/main/examples/data_connectors).


```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/vector_stores/SimpleIndexDemo.ipynb
../../examples/vector_stores/RedisIndexDemo.ipynb
../../examples/vector_stores/QdrantIndexDemo.ipynb
../../examples/vector_stores/FaissIndexDemo.ipynb
../../examples/vector_stores/DeepLakeIndexDemo.ipynb
../../examples/vector_stores/MyScaleIndexDemo.ipynb
../../examples/vector_stores/MetalIndexDemo.ipynb
../../examples/vector_stores/WeaviateIndexDemo.ipynb
../../examples/vector_stores/OpensearchDemo.ipynb
../../examples/vector_stores/PineconeIndexDemo.ipynb
../../examples/vector_stores/ChromaIndexDemo.ipynb
../../examples/vector_stores/LanceDBIndexDemo.ipynb
../../examples/vector_stores/MilvusIndexDemo.ipynb
../../examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb
../../examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb
../../examples/vector_stores/AsyncIndexCreationDemo.ipynb
```
