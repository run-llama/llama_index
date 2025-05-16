# Using Vector Stores

LlamaIndex offers multiple integration points with vector stores / vector databases:

1. LlamaIndex can use a vector store itself as an index. Like any other index, this index can store documents and be used to answer queries.
2. LlamaIndex can load data from vector stores, similar to any other data connector. This data can then be used within LlamaIndex data structures.

## Using a Vector Store as an Index

LlamaIndex also supports different vector stores
as the storage backend for `VectorStoreIndex`.

- Alibaba Cloud OpenSearch (`AlibabaCloudOpenSearchStore`). [QuickStart](https://help.aliyun.com/zh/open-search/vector-search-edition).
- Amazon Neptune - Neptune Analytics (`NeptuneAnalyticsVectorStore`). [Working with vector similarity in Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vector-similarity.html).
- Apache Cassandra® and Astra DB through CQL (`CassandraVectorStore`). [Installation](https://cassandra.apache.org/doc/stable/cassandra/getting_started/installing.html) [Quickstart](https://docs.datastax.com/en/astra-serverless/docs/vector-search/overview.html)
- Astra DB (`AstraDBVectorStore`). [Quickstart](https://docs.datastax.com/en/astra/home/astra.html).
- AWS Document DB (`AWSDocDbVectorStore`). [Quickstart](https://docs.aws.amazon.com/documentdb/latest/developerguide/get-started-guide.html).
- Azure AI Search (`AzureAISearchVectorStore`). [Quickstart](https://learn.microsoft.com/en-us/azure/search/search-get-started-vector)
- Chroma (`ChromaVectorStore`) [Installation](https://docs.trychroma.com/getting-started)
- ClickHouse (`ClickHouseVectorStore`) [Installation](https://clickhouse.com/docs/en/install)
- Couchbase (`CouchbaseSearchVectorStore`) [Installation](https://www.couchbase.com/products/capella/)
- DashVector (`DashVectorStore`). [Installation](https://help.aliyun.com/document_detail/2510230.html).
- DeepLake (`DeepLakeVectorStore`) [Installation](https://docs.deeplake.ai/en/latest/Installation.html)
- DocArray (`DocArrayHnswVectorStore`, `DocArrayInMemoryVectorStore`). [Installation/Python Client](https://github.com/docarray/docarray#installation).
- Elasticsearch (`ElasticsearchStore`) [Installation](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)
- Epsilla (`EpsillaVectorStore`) [Installation/Quickstart](https://epsilla-inc.gitbook.io/epsilladb/quick-start)
- Faiss (`FaissVectorStore`). [Installation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).
- Google AlloyDB for PostgreSQL (`AlloyDBVectorStore`). [QuickStart](https://github.com/googleapis/llama-index-alloydb-pg-python/blob/main/samples/llama_index_vector_store.ipynb).
- Google Cloud SQL for PostgreSQL (`PostgresVectorStore`). [Quickstart](https://github.com/googleapis/llama-index-cloud-sql-pg-python/blob/main/samples/llama_index_vector_store.ipynb)
- Hnswlib (`HnswlibVectorStore`). [Installation](https://github.com/nmslib/hnswlib?tab=readme-ov-file#bindings-installation).
- txtai (`TxtaiVectorStore`). [Installation](https://neuml.github.io/txtai/install/).
- Jaguar (`JaguarVectorStore`). [Installation](http://www.jaguardb.com/docsetup.html).
- Lantern (`LanternVectorStore`). [Quickstart](https://docs.lantern.dev/get-started/overview).
- MariaDB (`MariaDBVectorStore`). [MariaDB Vector Overview](https://mariadb.com/kb/en/vector-overview/)
- Milvus (`MilvusVectorStore`). [Installation](https://milvus.io/docs)
- MongoDB Atlas (`MongoDBAtlasVectorSearch`). [Installation/Quickstart](https://www.mongodb.com/atlas/database).
- MyScale (`MyScaleVectorStore`). [Quickstart](https://docs.myscale.com/en/quickstart/). [Installation/Python Client](https://docs.myscale.com/en/python-client/).
- Neo4j (`Neo4jVectorIndex`). [Installation](https://neo4j.com/docs/operations-manual/current/installation/).
- OceanBase (`OceanBaseVectorStore`). [OceanBase Overview](https://github.com/oceanbase/oceanbase). [Quickstart](../../examples/vector_stores/OceanBaseVectorStore.ipynb). [Python Client](https://github.com/oceanbase/pyobvector)
- Opensearch (`OpensearchVectorStore`) [Opensearch as vector database](https://opensearch.org/platform/search/vector-database.html). [QuickStart](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- Pinecone (`PineconeVectorStore`). [Installation/Quickstart](https://docs.pinecone.io/docs/quickstart).
- Qdrant (`QdrantVectorStore`) [Installation](https://qdrant.tech/documentation/install/) [Python Client](https://qdrant.tech/documentation/install/#python-client)
- LanceDB (`LanceDBVectorStore`) [Installation/Quickstart](https://lancedb.github.io/lancedb/basic/)
- Redis (`RedisVectorStore`). [Installation](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/).
- Relyt (`RelytVectorStore`). [Quickstart](https://docs.relyt.cn/docs/vector-engine/).
- Supabase (`SupabaseVectorStore`). [Quickstart](https://supabase.github.io/vecs/api/).
- Tablestore (`Tablestore`). [Tablestore Overview](https://www.aliyun.com/product/ots). [Quickstart](../../examples/vector_stores/TablestoreDemo.ipynb). [Python Client](https://github.com/aliyun/aliyun-tablestore-python-sdk).
- TiDB (`TiDBVectorStore`). [Quickstart](../../examples/vector_stores/TiDBVector.ipynb). [Installation](https://tidb.cloud/ai). [Python Client](https://github.com/pingcap/tidb-vector-python).
- TimeScale (`TimescaleVectorStore`). [Installation](https://github.com/timescale/python-vector).
- Upstash (`UpstashVectorStore`). [Quickstart](https://upstash.com/docs/vector/overall/getstarted)
- Vertex AI Vector Search (`VertexAIVectorStore`). [Quickstart](https://cloud.google.com/vertex-ai/docs/vector-search/quickstart)
- Weaviate (`WeaviateVectorStore`). [Installation](https://weaviate.io/developers/weaviate/installation). [Python Client](https://weaviate.io/developers/weaviate/client-libraries/python).
- WordLift (`WordliftVectorStore`). [Quickstart](https://docs.wordlift.io/llm-connectors/wordlift-vector-store/). [Python Client](https://pypi.org/project/wordlift-client/).
- Zep (`ZepVectorStore`). [Installation](https://docs.getzep.com/deployment/quickstart/). [Python Client](https://docs.getzep.com/sdk/).
- Zilliz (`MilvusVectorStore`). [Quickstart](https://zilliz.com/doc/quick_start)

A detailed API reference is [found here](../../api_reference/storage/vector_store/index.md).

Similar to any other index within LlamaIndex (tree, keyword table, list), `VectorStoreIndex` can be constructed upon any collection
of documents. We use the vector store within the index to store embeddings for the input text chunks.

Once constructed, the index can be used for querying.

**Default Vector Store Index Construction/Querying**

By default, `VectorStoreIndex` uses an in-memory `SimpleVectorStore`
that's initialized as part of the default storage context.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents and build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

**Custom Vector Store Index Construction/Querying**

We can query over a custom vector store as follows:

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store=DeepLakeVectorStore(dataset_path="<dataset_path>")
)

# Load documents and build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

Below we show more examples of how to construct various vector stores we support.

**Alibaba Cloud OpenSearch**

```python
from llama_index.vector_stores.alibabacloud_opensearch import (
    AlibabaCloudOpenSearchStore,
    AlibabaCloudOpenSearchConfig,
)

config = AlibabaCloudOpenSearchConfig(
    endpoint="***",
    instance_id="***",
    username="your_username",
    password="your_password",
    table_name="llama",
)

vector_store = AlibabaCloudOpenSearchStore(config)
```

**Google AlloyDB for PostgreSQL**

```bash
pip install llama-index
pip install llama-index-alloydb-pg
pip install llama-index-llms-vertex
gcloud services enable aiplatform.googleapis.com
```

```python
from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore
from llama_index.core import Settings
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
import google.auth

# Replace with your own AlloyDB info
engine = AlloyDBEngine.from_instance(
    project_id=PROJECT_ID,
    region=REGION,
    cluster=CLUSTER,
    instance=INSTANCE,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
)

engine.init_vector_store_table(
    table_name=TABLE_NAME,
    vector_size=768,  # Vector size for VertexAI model(textembedding-gecko@latest)
)

vector_store = AlloyDBVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
)
```

**Amazon Neptune - Neptune Analytics**

```python
from llama_index.vector_stores.neptune import NeptuneAnalyticsVectorStore

graph_identifier = ""
embed_dim = 1536

neptune_vector_store = NeptuneAnalyticsVectorStore(
    graph_identifier=graph_identifier, embedding_dimension=1536
)
```

**Apache Cassandra®**

```python
from llama_index.vector_stores.cassandra import CassandraVectorStore
import cassio

# To use an Astra DB cloud instance through CQL:
cassio.init(database_id="1234abcd-...", token="AstraCS:...")

# For a Cassandra cluster:
from cassandra.cluster import Cluster

cluster = Cluster(["127.0.0.1"])
cassio.init(session=cluster.connect(), keyspace="my_keyspace")

# After the above `cassio.init(...)`, create a vector store:
vector_store = CassandraVectorStore(
    table="cass_v_table", embedding_dimension=1536
)
```

**Astra DB**

```python
from llama_index.vector_stores.astra_db import AstraDBVectorStore

astra_db_store = AstraDBVectorStore(
    token="AstraCS:xY3b...",  # Your Astra DB token
    api_endpoint="https://012...abc-us-east1.apps.astra.datastax.com",  # Your Astra DB API endpoint
    collection_name="astra_v_table",  # Table name of your choice
    embedding_dimension=1536,  # Embedding dimension of the embeddings model used
)
```

**Azure Cognitive Search**

```python
from azure.core.credentials import AzureKeyCredential
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore

search_service_api_key = "YOUR-AZURE-SEARCH-SERVICE-ADMIN-KEY"
search_service_endpoint = "YOUR-AZURE-SEARCH-SERVICE-ENDPOINT"
search_service_api_version = "2023-11-01"
credential = AzureKeyCredential(search_service_api_key)

# Index name to use
index_name = "llamaindex-vector-demo"

client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=credential,
)

vector_store = AzureAISearchVectorStore(
    search_or_index_client=client,
    index_name=index_name,
    embedding_dimensionality=1536,
)
```

**Chroma**

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Creating a Chroma client
# EphemeralClient operates purely in-memory, PersistentClient will also save to disk
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)
```

**ClickHouse**

```python
import clickhouse_connect
from llama_index.vector_stores import ClickHouseVectorStore

# Creating a ClickHouse client
client = clickhouse_connect.get_client(
    host="YOUR_CLUSTER_HOST",
    port=8123,
    username="YOUR_USERNAME",
    password="YOUR_CLUSTER_PASSWORD",
)

# construct vector store
vector_store = ClickHouseVectorStore(clickhouse_client=client)
```

**Couchbase**

```python
from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

# Create a Couchbase Cluster object
auth = PasswordAuthenticator("DATABASE_USERNAME", "DATABASE_PASSWORD")
options = ClusterOptions(auth)
cluster = Cluster("CLUSTER_CONNECTION_STRING", options)

# Wait until the cluster is ready for use.
cluster.wait_until_ready(timedelta(seconds=5))

# Create the Vector Store
vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name="BUCKET_NAME",
    scope_name="SCOPE_NAME",
    collection_name="COLLECTION_NAME",
    index_name="SEARCH_INDEX_NAME",
)
```

**DashVector**

```python
import dashvector
from llama_index.vector_stores.dashvector import DashVectorStore

# init dashvector client
client = dashvector.Client(
    api_key="your-dashvector-api-key",
    endpoint="your-dashvector-cluster-endpoint",
)

# creating a DashVector collection
client.create("quickstart", dimension=1536)
collection = client.get("quickstart")

# construct vector store
vector_store = DashVectorStore(collection)
```

**DeepLake**

```python
import os
import getpath
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

os.environ["OPENAI_API_KEY"] = getpath.getpath("OPENAI_API_KEY: ")
os.environ["ACTIVELOOP_TOKEN"] = getpath.getpath("ACTIVELOOP_TOKEN: ")
dataset_path = "hub://adilkhan/paul_graham_essay"

# construct vector store
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
```

**DocArray**

```python
from llama_index.vector_stores.docarray import (
    DocArrayHnswVectorStore,
    DocArrayInMemoryVectorStore,
)

# construct vector store
vector_store = DocArrayHnswVectorStore(work_dir="hnsw_index")

# alternatively, construct the in-memory vector store
vector_store = DocArrayInMemoryVectorStore()
```

**Elasticsearch**

First, you can start Elasticsearch either locally or on [Elastic cloud](https://cloud.elastic.co/registration?utm_source=llama-index&utm_content=documentation).

To start Elasticsearch locally with docker, run the following command:

```bash
docker run -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/elasticsearch/elasticsearch:8.9.0
```

Then connect and use Elasticsearch as a vector database with LlamaIndex

```python
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

vector_store = ElasticsearchStore(
    index_name="llm-project",
    es_url="http://localhost:9200",
    # Cloud connection options:
    # es_cloud_id="<cloud_id>",
    # es_user="elastic",
    # es_password="<password>",
)
```

This can be used with the `VectorStoreIndex` to provide a query interface for retrieval, querying, deleting, persisting the index, and more.

**Epsilla**

```python
from pyepsilla import vectordb
from llama_index.vector_stores.epsilla import EpsillaVectorStore

# Creating an Epsilla client
epsilla_client = vectordb.Client()

# Construct vector store
vector_store = EpsillaVectorStore(client=epsilla_client)
```

**Note**: `EpsillaVectorStore` depends on the `pyepsilla` library and a running Epsilla vector database.
Use `pip/pip3 install pyepsilla` if not installed yet.
A running Epsilla vector database could be found through docker image.
For complete instructions, see the following documentation:
https://epsilla-inc.gitbook.io/epsilladb/quick-start

**Faiss**

```python
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

# create faiss index
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# construct vector store
vector_store = FaissVectorStore(faiss_index)

# if update/delete functionality is needed you can leverage the FaissMapVectorStore

d = 1536
faiss_index = faiss.IndexFlatL2(d)
id_map_index = faiss.IndexIDMap2(faiss_index)
vector_store = FaissMapVectorStore(id_map_index)

...

# NOTE: since faiss index is in-memory, we need to explicitly call
#       vector_store.persist() or storage_context.persist() to save it to disk.
#       persist() takes in optional arg persist_path. If none give, will use default paths.
storage_context.persist()
```

**Google Cloud SQL for PostgreSQL**

```bash
pip install llama-index
pip install llama-index-cloud-sql-pg
pip install llama-index-llms-vertex
gcloud services enable aiplatform.googleapis.com
```

```python
from llama_index_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from llama_index.core import Settings
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
import google.auth

# Replace with your own Cloud SQL info
engine = PostgresEngine.from_instance(
    project_id=PROJECT_ID,
    region=REGION,
    instance=INSTANCE,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
)

engine.init_vector_store_table(
    table_name=TABLE_NAME,
    vector_size=768,  # Vector size for VertexAI model(textembedding-gecko@latest)
)

vector_store = PostgresVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
)
```

**txtai**

```python
import txtai
from llama_index.vector_stores.txtai import TxtaiVectorStore

# create txtai index
txtai_index = txtai.ann.ANNFactory.create(
    {"backend": "numpy", "dimension": 512}
)

# construct vector store
vector_store = TxtaiVectorStore(txtai_index)
```

**Jaguar**

```python
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from jaguardb_http_client.JaguarHttpClient import JaguarHttpClient
from llama_index.vector_stores.jaguar import JaguarVectorStore


# construct vector store client
url = "http://127.0.0.1:8080/fwww/"
pod = "vdb"
store = "llamaindex_rag_store"
vector_index = "v"
vector_type = "cosine_fraction_float"
vector_dimension = 3

# require JAGUAR_API_KEY environment variable or file $HOME/.jagrc to hold the
# jaguar API key to connect to jaguar store server
vector_store = JaguarVectorStore(
    pod, store, vector_index, vector_type, vector_dimension, url
)

# login to jaguar server for security authentication
vector_store.login()

# create a vector store on the back-end server
metadata_fields = "author char(32), category char(16)"
text_size = 1024
vector_store.create(metadata_fields, text_size)

# store some text
node = TextNode(
    text="Return of King Lear",
    metadata={"author": "William", "category": "Tragedy"},
    embedding=[0.9, 0.1, 0.4],
)
vector_store.add(nodes=[node], use_node_metadata=True)

# make a query
qembedding = [0.4, 0.2, 0.8]
vsquery = VectorStoreQuery(query_embedding=qembedding, similarity_top_k=1)
query_result = vector_store.query(vsquery)

# make a query with metadata filter (where condition)
qembedding = [0.6, 0.1, 0.4]
vsquery = VectorStoreQuery(query_embedding=qembedding, similarity_top_k=3)
where = "author='Eve' or (author='Adam' and category='History')"
query_result = vector_store.query(vsquery, where=where)

# make a query ignoring old data (with time cutoff)
qembedding = [0.3, 0.3, 0.8]
vsquery = VectorStoreQuery(query_embedding=qembedding, similarity_top_k=3)
args = "day_cutoff=180"  # only search recent 180 days data
query_result = vector_store.query(vsquery, args=args)

# check if a vector is anomalous
text = ("Gone With The Wind",)
embed_of_text = [0.7, 0.1, 0.2]
node = TextNode(text=text, embedding=embed_of_text)
true_or_false = vector_store.is_anomalous(node)

# llama_index RAG application
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

question = "What did the author do growing up?"

storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model = OpenAIEmbedding()
embed_of_question = [0.7, 0.1, 0.2]

db_documents = vector_store.load_documents(embed_of_question, 10)
index = VectorStoreIndex.from_documents(
    db_documents,
    embed_model=embed_model,
    storage_context=storage_context,
)

query_engine = index.as_query_engine()
print(f"Question: {question}")
response = query_engine.query(question)
print(f"Answer: {str(response)}")

# logout to clean up resources
vector_store.logout()
```

**Note**: Client(requires jaguardb-http-client) <--> Http Gateway <--> JaguarDB Server
Client side needs to run: "pip install -U jaguardb-http-client"

**MariaDB**

```python
from llama_index.vector_stores.mariadb import MariaDBVectorStore

vector_store = MariaDBVectorStore.from_params(
    host="localhost",
    port=3306,
    user="llamaindex",
    password="password",
    database="vectordb",
    table_name="llama_index_vectorstore",
    embed_dim=1536,  # OpenAI embedding dimension
)
```

**Milvus**

- Milvus Index offers the ability to store both Documents and their embeddings.

```python
import pymilvus
from llama_index.vector_stores.milvus import MilvusVectorStore

# construct vector store
vector_store = MilvusVectorStore(
    uri="https://localhost:19530", overwrite="True"
)
```

**Note**: `MilvusVectorStore` depends on the `pymilvus` library.
Use `pip install pymilvus` if not already installed.
If you get stuck at building wheel for `grpcio`, check if you are using python 3.11
(there's a known issue: https://github.com/milvus-io/pymilvus/issues/1308)
and try downgrading.

**MongoDBAtlas**

```python
# Provide URI to constructor, or use environment variable
import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader

# mongo_uri = os.environ["MONGO_URI"]
mongo_uri = (
    "mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority"
)
mongodb_client = pymongo.MongoClient(mongo_uri)

# construct store
store = MongoDBAtlasVectorSearch(mongodb_client)
storage_context = StorageContext.from_defaults(vector_store=store)
uber_docs = SimpleDirectoryReader(
    input_files=["../data/10k/uber_2021.pdf"]
).load_data()

# construct index
index = VectorStoreIndex.from_documents(
    uber_docs, storage_context=storage_context
)
```

**MyScale**

```python
import clickhouse_connect
from llama_index.vector_stores.myscale import MyScaleVectorStore

# Creating a MyScale client
client = clickhouse_connect.get_client(
    host="YOUR_CLUSTER_HOST",
    port=8443,
    username="YOUR_USERNAME",
    password="YOUR_CLUSTER_PASSWORD",
)


# construct vector store
vector_store = MyScaleVectorStore(myscale_client=client)
```

**Neo4j**

- Neo4j stores texts, metadata, and embeddings and can be customized to return graph data in the form of metadata.

```python
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

# construct vector store
neo4j_vector = Neo4jVectorStore(
    username="neo4j",
    password="pleaseletmein",
    url="bolt://localhost:7687",
    embed_dim=1536,
)
```

**Pinecone**

```python
import pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Creating a Pinecone index
api_key = "api_key"
pinecone.init(api_key=api_key, environment="us-west1-gcp")
pinecone.create_index(
    "quickstart", dimension=1536, metric="euclidean", pod_type="p1"
)
index = pinecone.Index("quickstart")

# construct vector store
vector_store = PineconeVectorStore(pinecone_index=index)
```

**Qdrant**

```python
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Creating a Qdrant vector store
client = qdrant_client.QdrantClient(
    host="<qdrant-host>", api_key="<qdrant-api-key>", https=True
)
collection_name = "paul_graham"

# construct vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
)
```

**Redis**

First, start Redis-Stack (or get url from Redis provider)

```bash
docker run --name redis-vecdb -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

Then connect and use Redis as a vector database with LlamaIndex

```python
from llama_index.vector_stores.redis import RedisVectorStore

vector_store = RedisVectorStore(
    index_name="llm-project",
    redis_url="redis://localhost:6379",
    overwrite=True,
)
```

This can be used with the `VectorStoreIndex` to provide a query interface for retrieval, querying, deleting, persisting the index, and more.

**SingleStore**

```python
from llama_index.vector_stores.singlestoredb import SingleStoreVectorStore
import os

# can set the singlestore db url in env
# or pass it in as an argument to the SingleStoreVectorStore constructor
os.environ["SINGLESTOREDB_URL"] = "PLACEHOLDER URL"
vector_store = SingleStoreVectorStore(
    table_name="embeddings",
    content_field="content",
    metadata_field="metadata",
    vector_field="vector",
    timeout=30,
)
```

**Tablestore**

```python
import tablestore
from llama_index.vector_stores.tablestore import TablestoreVectorStore

# create a vector store that does not support filtering non-vector fields
simple_vector_store = TablestoreVectorStore(
    endpoint="<end_point>",
    instance_name="<instance_name>",
    access_key_id="<access_key_id>",
    access_key_secret="<access_key_secret>",
    vector_dimension=512,
)

# create a vector store that support filtering non-vector fields
vector_store_with_meta_data = TablestoreVectorStore(
    endpoint="<end_point>",
    instance_name="<instance_name>",
    access_key_id="<access_key_id>",
    access_key_secret="<access_key_secret>",
    vector_dimension=512,
    # optional: custom metadata mapping is used to filter non-vector fields.
    metadata_mappings=[
        tablestore.FieldSchema(
            "type",  # non-vector fields
            tablestore.FieldType.KEYWORD,
            index=True,
            enable_sort_and_agg=True,
        ),
        tablestore.FieldSchema(
            "time",  # non-vector fields
            tablestore.FieldType.LONG,
            index=True,
            enable_sort_and_agg=True,
        ),
    ],
)
```

**TiDB**

```python
from llama_index.vector_stores.tidbvector import TiDBVectorStore

tidbvec = TiDBVectorStore(
    # connection url format
    # - mysql+pymysql://root@34.212.137.91:4000/test
    connection_string="PLACEHOLDER URL",
    table_name="llama_index_vectorstore",
    distance_strategy="cosine",
    vector_dimension=1536,
)
```

**Timescale**

```python
from llama_index.vector_stores.timescalevector import TimescaleVectorStore

vector_store = TimescaleVectorStore.from_params(
    service_url="YOUR TIMESCALE SERVICE URL",
    table_name="paul_graham_essay",
)
```

**Upstash**

```python
from llama_index.vector_stores.upstash import UpstashVectorStore

vector_store = UpstashVectorStore(url="YOUR_URL", token="YOUR_TOKEN")
```

**Vertex AI Vector Search**

```python
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

vector_store = VertexAIVectorStore(
    project_id="[your-google-cloud-project-id]",
    region="[your-google-cloud-region]",
    index_id="[your-index-resource-name]",
    endpoint_id="[your-index-endpoint-name]",
)
```

**Weaviate**

```python
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# creating a Weaviate client
resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    "https://<cluster-id>.semi.network/",
    auth_client_secret=resource_owner_config,
)

# construct vector store
vector_store = WeaviateVectorStore(weaviate_client=client)
```

**Zep**

Zep stores texts, metadata, and embeddings. All are returned in search results.

```python
from llama_index.vector_stores.zep import ZepVectorStore

vector_store = ZepVectorStore(
    api_url="<api_url>",
    api_key="<api_key>",
    collection_name="<unique_collection_name>",  # Can either be an existing collection or a new one
    embedding_dimensions=1536,  # Optional, required if creating a new collection
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Query index using both a text query and metadata filters
filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)
retriever = index.as_retriever(filters=filters)
result = retriever.retrieve("What is inception about?")
```

**Zilliz**

- Zilliz Cloud (hosted version of Milvus) uses the Milvus Index with some extra arguments.

```python
import pymilvus
from llama_index.vector_stores.milvus import MilvusVectorStore


# construct vector store
vector_store = MilvusVectorStore(
    uri="foo.vectordb.zillizcloud.com",
    token="your_token_here",
    overwrite="True",
)
```

[Example notebooks can be found here](https://github.com/jerryjliu/llama_index/tree/main/docs/docs/examples/vector_stores).

## Loading Data from Vector Stores using Data Connector

LlamaIndex supports loading data from a huge number of sources. See [Data Connectors](../../module_guides/loading/connector/modules.md) for more details and API documentation.

AlloyDB stores both document and vectors.
This tutorial demonstrates the synchronous interface. All synchronous methods have corresponding asynchronous methods.
This is an example of how to use AlloyDB:

```bash
pip install llama-index
pip install llama-index-alloydb-pg
```

```python
from llama_index.core import SummaryIndex
from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBReader

engine = AlloyDBEngine.from_instance(
    project_id=PROJECT_ID,
    region=REGION,
    cluster=CLUSTER,
    instance=INSTANCE,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
)
reader = AlloyDBReader.create_sync(
    engine,
    table_name=TABLE_NAME,
)
documents = reader.load_data()

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
display(Markdown(f"<b>{response}</b>"))
```

Google Cloud SQL for PostgreSQL stores both document and vectors.
This tutorial demonstrates the synchronous interface. All synchronous methods have corresponding asynchronous methods.
This is an example of how to use Cloud SQL for PostgreSQL:

```bash
pip install llama-index
pip install llama-index-cloud-sql-pg
```

```python
from llama_index.core import SummaryIndex
from llama_index_cloud_sql_pg import PostgresEngine, PostgresReader

engine = PostgresEngine.from_instance(
    project_id=PROJECT_ID,
    region=REGION,
    instance=INSTANCE,
    database=DATABASE,
    user=USER,
    password=PASSWORD,
)
reader = PostgresReader.create_sync(
    engine,
    table_name=TABLE_NAME,
)
documents = reader.load_data()

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
display(Markdown(f"<b>{response}</b>"))
```

Chroma stores both documents and vectors. This is an example of how to use Chroma:

```python
from llama_index.readers.chroma import ChromaReader
from llama_index.core import SummaryIndex

# The chroma reader loads data from a persisted Chroma collection.
# This requires a collection name and a persist directory.
reader = ChromaReader(
    collection_name="chroma_collection",
    persist_directory="examples/data_connectors/chroma_collection",
)

query_vector = [n1, n2, n3, ...]

documents = reader.load_data(
    collection_name="demo", query_vector=query_vector, limit=5
)
index = SummaryIndex.from_documents(documents)

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

documents = reader.load_data(
    collection_name="demo", query_vector=query_vector, limit=5
)
```

NOTE: Since Weaviate can store a hybrid of document and vector objects, the user may either choose to explicitly specify `class_name` and `properties` in order to query documents, or they may choose to specify a raw GraphQL query. See below for usage.

```python
# option 1: specify class_name and properties

# 1) load data using class_name and properties
documents = reader.load_data(
    class_name="<class_name>",
    properties=["property1", "property2", "..."],
    separate_documents=True,
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

query_vector = [n1, n2, n3, ...]

documents = reader.load_data(
    index_name="quickstart",
    id_to_text_map=id_to_text_map,
    top_k=3,
    vector=query_vector,
    separate_documents=True,
)
```

[Example notebooks can be found here](https://github.com/jerryjliu/llama_index/tree/main/docs/docs/examples/data_connectors).

## Vector Store Examples

- [Alibaba Cloud OpenSearch](../../examples/vector_stores/AlibabaCloudOpenSearchIndexDemo.ipynb)
- [Amazon Neptune - Neptune Analytics](../../examples/vector_stores/AmazonNeptuneVectorDemo.ipynb)
- [Astra DB](../../examples/vector_stores/AstraDBIndexDemo.ipynb)
- [Async Index Creation](../../examples/vector_stores/AsyncIndexCreationDemo.ipynb)
- [Azure AI Search](../../examples/vector_stores/AzureAISearchIndexDemo.ipynb)
- [Azure Cosmos DB](../../examples/vector_stores/AzureCosmosDBMongoDBvCoreDemo.ipynb)
- [Caasandra](../../examples/vector_stores/CassandraIndexDemo.ipynb)
- [Chromadb](../../examples/vector_stores/ChromaIndexDemo.ipynb)
- [Couchbase](../../examples/vector_stores/CouchbaseVectorStoreDemo.ipynb)
- [Dash](../../examples/vector_stores/DashvectorIndexDemo.ipynb)
- [Deeplake](../../examples/vector_stores/DeepLakeIndexDemo.ipynb)
- [DocArray HNSW](../../examples/vector_stores/DocArrayHnswIndexDemo.ipynb)
- [DocArray in-Memory](../../examples/vector_stores/DocArrayInMemoryIndexDemo.ipynb)
- [Espilla](../../examples/vector_stores/EpsillaIndexDemo.ipynb)
- [Google AlloyDB for PostgreSQL](../../examples/vector_stores/AlloyDBVectorStoreDemo.ipynb)
- [Google Cloud SQL for PostgreSQL](../../examples/vector_stores/CloudSQLPgVectorStoreDemo.ipynb)
- [LanceDB](../../examples/vector_stores/LanceDBIndexDemo.ipynb)
- [Lantern](../../examples/vector_stores/LanternIndexDemo.ipynb)
- [Metal](../../examples/vector_stores/MetalIndexDemo.ipynb)
- [Milvus](../../examples/vector_stores/MilvusIndexDemo.ipynb)
- [Milvus Async API](../../examples/vector_stores/MilvusAsyncAPIDemo.ipynb)
- [Milvus Full-Text Search](../../examples/vector_stores/MilvusFullTextSearchDemo.ipynb)
- [Milvus Hybrid Search](../../examples/vector_stores/MilvusHybridIndexDemo.ipynb)
- [MyScale](../../examples/vector_stores/MyScaleIndexDemo.ipynb)
- [ElsaticSearch](../../examples/vector_stores/ElasticsearchIndexDemo.ipynb)
- [FAISS](../../examples/vector_stores/FaissIndexDemo.ipynb)
- [MongoDB Atlas](../../examples/vector_stores/MongoDBAtlasVectorSearch.ipynb)
- [Neo4j](../../examples/vector_stores/Neo4jVectorDemo.ipynb)
- [OpenSearch](../../examples/vector_stores/OpensearchDemo.ipynb)
- [Pinecone](../../examples/vector_stores/PineconeIndexDemo.ipynb)
- [Pinecone Hybrid Search](../../examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb)
- [PGvectoRS](../../examples/vector_stores/PGVectoRsDemo.ipynb)
- [Postgres](../../examples/vector_stores/postgres.ipynb)
- [Redis](../../examples/vector_stores/RedisIndexDemo.ipynb)
- [Qdrant](../../examples/vector_stores/QdrantIndexDemo.ipynb)
- [Qdrant Hybrid Search](../../examples/vector_stores/qdrant_hybrid.ipynb)
- [Rockset](../../examples/vector_stores/RocksetIndexDemo.ipynb)
- [Simple](../../examples/vector_stores/SimpleIndexDemo.ipynb)
- [Supabase](../../examples/vector_stores/SupabaseVectorIndexDemo.ipynb)
- [Tablestore](../../examples/vector_stores/TablestoreDemo.ipynb)
- [Tair](../../examples/vector_stores/TairIndexDemo.ipynb)
- [Tencent](../../examples/vector_stores/TencentVectorDBIndexDemo.ipynb)
- [Timesacle](../../examples/vector_stores/Timescalevector.ipynb)
- [Upstash](../../examples/vector_stores/UpstashVectorDemo.ipynb)
- [Weaviate](../../examples/vector_stores/WeaviateIndexDemo.ipynb)
- [Weaviate Hybrid Search](../../examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb)
- [WordLift](../../examples/vector_stores/WordLiftDemo.ipynb)
- [Zep](../../examples/vector_stores/ZepIndexDemo.ipynb)
