使用向量存储

LlamaIndex提供多个与向量存储/向量数据库的集成点：

1. LlamaIndex可以将向量存储本身用作索引。与其他索引一样，此索引可以存储文档并用于回答查询。
2. LlamaIndex可以从向量存储中加载数据，类似于任何其他数据连接器。然后可以在LlamaIndex数据结构中使用此数据。

（vector-store-index）=

## 使用向量存储作为索引

LlamaIndex还支持不同的向量存储作为`GPTVectorStoreIndex`的存储后端。

- Chroma（`ChromaReader`）[安装](https://docs.trychroma.com/getting-started)
- DeepLake（`DeepLakeReader`）[安装](https://docs.deeplake.ai/en/latest/Installation.html)
- Qdrant（`QdrantReader`）[安装](https://qdrant.tech/documentation/install/) [Python Client](https://qdrant.tech/documentation/install/#python-client)
- Weaviate（`WeaviateReader`）。[安装](https://weaviate.io/developers/weaviate/installation)。[Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)。
- Pinecone（`PineconeReader`）。[安装/快速入门](https://docs.pinecone.io/docs/quickstart)。
- Faiss（`FaissReader`）。[安装](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)。
- Milvus（`MilvusReader`）。[安装](https://milvus.io/docs)
- Zilliz（`MilvusReader`）。[快速入门](https://zilliz.com/doc/quick_start)
- MyScale（`MyScaleReader`）。[快速入门](https://docs.myscale.com/en/quickstart/)。[安装/Python Client](https://docs.myscale.com/en/python-client/)。

详细的API参考[在这里](/reference/indices/vector_store.rst)。

与LlamaIndex中的任何其他索引（树，关键字表，列表）一样，可以在任何文档集合上构建`GPTVectorStoreIndex`。我们在索引中使用向量存储来存储输入文本块的嵌入。

一旦构建完成，就可以使用索引进行查询。

**默认向量存储索引构建/查询**

默认情况下，`GPTVectorStoreIndex`使用作为默认存储上下文的一部分初始化的内存`SimpleVectorStore`。

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

# 加载文档并构建索引
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# 查询索引
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

```

**自定义向量存储索引构建/查询**

我们可以在自定义向量存储如下：

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import DeepLakeVectorStore

# 构建向量存储并自定义存储上下文
storage_context = StorageContext.from_defaults(
    vector_store = DeepLakeVectorStore(dataset_path="<dataset_path>")
)

# 加载文档并构建索引
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 查询索引
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

以下是构建各种支持的向量存储的更多示例。

**Redis**
首先，启动Redis-Stack（或从Redis提供商获取url）

```bash
docker run --name redis-vecdb -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

然后连接并使用Redis作为LlamaIndex的向量数据库

```python
from llama_index.vector_stores import RedisVectorStore
vector_store = RedisVectorStore(
    index_name="llm-project",
    redis_url="redis://localhost:6379",
    overwrite=True
)
```

这可以与`GPTVectorStoreIndex`一起使用，为检索，查询，删除，持久化索引等提供查询界面。

**DeepLake**

```python
import os
import getpath
from llama_index.vector_stores import DeepLakeVectorStore

os.environ["OPENAI_API_KEY"] = getpath.getpath("OPENAI_API_KEY: ")
os.environ["ACTIVELOOP_TOKEN"] = getpath.getpath("ACTIVELOOP_TOKEN: ")
dataset_path = "hub://adilkhan/paul_graham_essay"

# 构建向量存储
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
```

**Faiss**

```python
import faiss
from llama_index.vector_stores import FaissVectorStore

# 创建faiss索引
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# 构建向量存储
vector_store = FaissVectorStore(faiss_index)

...

# 注意：由于faiss索引是内存中的，因此我们需要显式调用
#       vector_store.persist()或storage_context.persist()将其保存到磁盘。
#       persist（）接受可选参数persist_path。如果没有给出，将使用默认路径。
storage_context.persist()
```

**Weaviate**

```python
import weaviate
from llama_index.vector_stores import WeaviateVectorStore

# 创建Weaviate客户端
resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    "https://<cluster-id>.semi.network/", auth_client_secret=resource_owner_config
)

# 构建
vector_store = WeaviateVectorStore(client)
```**注意**：`MilvusVectorStore`依赖于`pymilvus`库。如果尚未安装，请使用`pip install pymilvus`。如果在构建`grpcio`的轮子时遇到困难，请检查是否使用了python 3.11（已知的问题：https://github.com/milvus-io/pymilvus/issues/1308），并尝试降级。

**Zilliz**

- Zilliz Cloud（Milvus的托管版本）使用Milvus Index，带有一些额外的参数。

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

**注意**：`MilvusVectorStore`LlamaIndex支持从以下来源加载数据。有关更多详细信息和API文档，请参阅[数据连接器]（/ how_to / data_connectors.md）。

Chroma存储文档和向量。这是如何使用Chroma的示例：

```python

from llama_index.readers.chroma import ChromaReader
from llama_index.indices import GPTListIndex

# Chroma读取器从持久化的Chroma集合中加载数据。
# 这需要一个集合名称和一个持久化目录。
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

[示例笔记本可以在这里找到]（https://github.com/jerryjliu/llama_index/tree/main/docs/examples/vector_stores）。

## 使用数据连接器从向量存储加载数据

LlamaIndex支持从以下来源加载数据。有关更多详细信息和API文档，请参阅[数据连接器]（/ how_to / data_connectors.md）。

Chroma存储文档和向量。这是如何使用Chroma的示例：

```python

from llama_index.readers.chroma import ChromaReader
from llama_index.indices import GPTListIndex

# Chroma读取器从持久化的Chroma集合中加载数据。
# 这需要一个集合名称和一个持久化目录。
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

Qdrant也存储文档和向量。这是如何使用Qdrant的示例：

```python

from llama_index.readers.qdrant import QdrantReader

reader = QdrantReader(host="localhost")

# query_vector是查询向量的嵌入表示
# 示例query_vector
# query_vector = [0.3, 0.3, 0.3, 0.3, ...]

query_vector = [n1, n2, n3, ...]

# 注意：必需的参数是collection_name，query_vector。
# 有关更多详细信息，请参阅Python客户端：https;//github.com/qdrant/qdrant_client

documents = reader.load_data(collection_name="demo", query_vector=query_vector, limit=5)

```

注意：由于Weaviate可以存储文档和向量对象的混合，因此用户可以选择显式指定`class_name`和`properties`以查询文档，或者可以选择指定原始GraphQL查询。有关用法，请参见下文。

```pyt# 选项1：指定class_name和属性

# 1）使用class_name和属性加载数据
documents = reader.load_data(
    class_name="<class_name>",
    properties=["property1", "property2", "..."],
    separate_documents=True
)

# 2）示例GraphQL查询
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

注意：Pinecone和Faiss数据加载器均假定相应的数据源仅存储向量；文本内容存储在其他地方。因此，两个数据加载器都要求用户在load_data调用中指定`id_to_text_map`。

例如，这是Pinecone数据加载器`PineconeReader`的示例用法：

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

[示例笔记本可在此处找到](https://github.com/jerryjliu/llama_index/tree/main/examples/data_connectors)。