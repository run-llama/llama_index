# Vector Stores

Vector stores contain embedding vectors of ingested document chunks
(and sometimes the document chunks as well).

## Simple Vector Store

By default, LlamaIndex uses a simple in-memory vector store that's great for quick experimentation.
They can be persisted to (and loaded from) disk by calling `vector_store.persist()` (and `SimpleVectorStore.from_persist_path(...)` respectively).

## Vector Store Options & Feature Support

LlamaIndex supports over 20 different vector store options.
We are actively adding more integrations and improving feature coverage for each.

| Vector Store             | Type                    | Metadata Filtering | Hybrid Search | Delete | Store Documents | Async |
|--------------------------|-------------------------| ------------------ | ------------- | ------ | --------------- | ----- |
| Alibaba Cloud OpenSearch | cloud                   | ✓                  |               | ✓      | ✓               | ✓     |
| Apache Cassandra®        | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| Astra DB                 | cloud                   | ✓                  |               | ✓      | ✓               |       |
| Azure AI Search          | cloud                   | ✓                  | ✓             | ✓      | ✓               |       |
| Azure CosmosDB MongoDB   | cloud                   |                    |               | ✓      | ✓               |       |
| BaiduVectorDB            | cloud                   | ✓                  | ✓             |        | ✓               |       |
| ChatGPT Retrieval Plugin | aggregator              |                    |               | ✓      | ✓               |       |
| Chroma                   | self-hosted             | ✓                  |               | ✓      | ✓               |       |
| Couchbase                | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |       |
| DashVector               | cloud                   | ✓                  | ✓             | ✓      | ✓               |       |
| Databricks               | cloud                   | ✓                  |               | ✓      | ✓               |       |
| Deeplake                 | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| DocArray                 | aggregator              | ✓                  |               | ✓      | ✓               |       |
| DuckDB                   | in-memory / self-hosted | ✓                  |               | ✓      | ✓               |       |
| DynamoDB                 | cloud                   |                    |               | ✓      |                 |       |
| Elasticsearch            | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓     |
| FAISS                    | in-memory               |                    |               |        |                 |       |
| Hnswlib                  | in-memory               |                    |               |        |                 |       |
| txtai                    | in-memory               |                    |               |        |                 |       |
| Jaguar                   | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |       |
| LanceDB                  | cloud                   | ✓                  |               | ✓      | ✓               |       |
| Lantern                  | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓     |
| Metal                    | cloud                   | ✓                  |               | ✓      | ✓               |       |
| MongoDB Atlas            | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| MyScale                  | cloud                   | ✓                  | ✓             | ✓      | ✓               |       |
| Milvus / Zilliz          | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |       |
| Neo4jVector              | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| OpenSearch               | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓     |
| Pinecone                 | cloud                   | ✓                  | ✓             | ✓      | ✓               |       |
| Postgres                 | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓     |
| pgvecto.rs               | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |       |
| Qdrant                   | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓     |
| Redis                    | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| Simple                   | in-memory               | ✓                  |               | ✓      |                 |       |
| SingleStore              | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| Supabase                 | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| Tablestore               | cloud                   | ✓                  | ✓             | ✓      | ✓               |       |
| Tair                     | cloud                   | ✓                  |               | ✓      | ✓               |       |
| TiDB                     | cloud                   | ✓                  |               | ✓      | ✓               |       |
| TencentVectorDB          | cloud                   | ✓                  | ✓             | ✓      | ✓               |       |
| Timescale                |                         | ✓                  |               | ✓      | ✓               | ✓     |
| Typesense                | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |       |
| Upstash                  | cloud                   |                    |               |        | ✓               |       |
| Vearch                   | self-hosted             | ✓                  |               | ✓      | ✓               |       |
| Vespa                    | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |       |
| Vertex AI Vector Search  | cloud                   | ✓                  |               | ✓      | ✓               |       |
| Weaviate                 | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |       |
| WordLift                 | cloud                   | ✓                  | ✓             | ✓      | ✓               | ✓     |

For more details, see [Vector Store Integrations](../../community/integrations/vector_stores.md).

## Example Notebooks

- [Alibaba Cloud OpenSearch](../../examples/vector_stores/AlibabaCloudOpenSearchIndexDemo.ipynb)
- [Astra DB](../../examples/vector_stores/AstraDBIndexDemo.ipynb)
- [Async Index Creation](../../examples/vector_stores/AsyncIndexCreationDemo.ipynb)
- [Azure AI Search](../../examples/vector_stores/AzureAISearchIndexDemo.ipynb)
- [Azure Cosmos DB](../../examples/vector_stores/AzureCosmosDBMongoDBvCoreDemo.ipynb)
- [Baidu](../../examples/vector_stores/BaiduVectorDBIndexDemo.ipynb)
- [Caasandra](../../examples/vector_stores/CassandraIndexDemo.ipynb)
- [Chromadb](../../examples/vector_stores/ChromaIndexDemo.ipynb)
- [Couchbase](../../examples/vector_stores/CouchbaseVectorStoreDemo.ipynb)
- [Dash](../../examples/vector_stores/DashvectorIndexDemo.ipynb)
- [Databricks](../../examples/vector_stores/DatabricksVectorSearchDemo.ipynb)
- [Deeplake](../../examples/vector_stores/DeepLakeIndexDemo.ipynb)
- [DocArray HNSW](../../examples/vector_stores/DocArrayHnswIndexDemo.ipynb)
- [DocArray in-Memory](../../examples/vector_stores/DocArrayInMemoryIndexDemo.ipynb)
- [DuckDB](../../examples/vector_stores/DuckDBDemo.ipynb)
- [Espilla](../../examples/vector_stores/EpsillaIndexDemo.ipynb)
- [Jaguar](../../examples/vector_stores/JaguarIndexDemo.ipynb)
- [LanceDB](../../examples/vector_stores/LanceDBIndexDemo.ipynb)
- [Lantern](../../examples/vector_stores/LanternIndexDemo.ipynb)
- [Metal](../../examples/vector_stores/MetalIndexDemo.ipynb)
- [Milvus](../../examples/vector_stores/MilvusIndexDemo.ipynb)
- [Milvus Hybrid Search](../../examples/vector_stores/MilvusHybridIndexDemo.ipynb)
- [MyScale](../../examples/vector_stores/MyScaleIndexDemo.ipynb)
- [ElasticSearch](../../examples/vector_stores/ElasticsearchIndexDemo.ipynb)
- [FAISS](../../examples/vector_stores/FaissIndexDemo.ipynb)
- [Hnswlib](../../examples/vector_stores/HnswlibIndexDemo.ipynb)
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
- [TiDB](../../examples/vector_stores/TiDBVector.ipynb)
- [Tencent](../../examples/vector_stores/TencentVectorDBIndexDemo.ipynb)
- [Timesacle](../../examples/vector_stores/Timescalevector.ipynb)
- [Upstash](../../examples/vector_stores/UpstashVectorDemo.ipynb)
- [Vearch](../../examples/vector_stores/VearchDemo.ipynb)
- [Vespa](../../examples/vector_stores/VespaIndexDemo.ipynb)
- [Vertex AI Vector Search](../../examples/vector_stores/VertexAIVectorSearchDemo.ipynb)
- [Weaviate](../../examples/vector_stores/WeaviateIndexDemo.ipynb)
- [Weaviate Hybrid Search](../../examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb)
- [WordLift](../../examples/vector_stores/WordLiftDemo.ipynb)
- [Zep](../../examples/vector_stores/ZepIndexDemo.ipynb)
