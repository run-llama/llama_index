# Vector Stores

Vector stores contain embedding vectors of ingested document chunks
(and sometimes the document chunks as well).

## Simple Vector Store

By default, LlamaIndex uses a simple in-memory vector store that's great for quick experimentation.
They can be persisted to (and loaded from) disk by calling `vector_store.persist()` (and `SimpleVectorStore.from_persist_path(...)` respectively).

## Vector Store Options & Feature Support

LlamaIndex supports over 20 different vector store options.
We are actively adding more integrations and improving feature coverage for each.

| Vector Store             | Type                | Metadata Filtering | Hybrid Search | Delete | Store Documents | Async |
| ------------------------ | ------------------- | ------------------ | ------------- | ------ | --------------- | ----- |
| Apache Cassandra®       | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| Astra DB                 | cloud               | ✓                  |               | ✓      | ✓               |       |
| Azure Cognitive Search   | cloud               |                    | ✓             | ✓      | ✓               |       |
| Azure CosmosDB MongoDB   | cloud               |                    |               | ✓      | ✓               |       |
| ChatGPT Retrieval Plugin | aggregator          |                    |               | ✓      | ✓               |       |
| Chroma                   | self-hosted         | ✓                  |               | ✓      | ✓               |       |
| DashVector               | cloud               | ✓                  |               | ✓      | ✓               |       |
| Deeplake                 | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| DocArray                 | aggregator          | ✓                  |               | ✓      | ✓               |       |
| DynamoDB                 | cloud               |                    |               | ✓      |                 |       |
| Elasticsearch            | self-hosted / cloud | ✓                  | ✓             | ✓      | ✓               | ✓     |
| FAISS                    | in-memory           |                    |               |        |                 |       |
| Jaguar                   | self-hosted / cloud | ✓                  | ✓             | ✓      | ✓               |       |
| LanceDB                  | cloud               | ✓                  |               | ✓      | ✓               |       |
| Lantern                  | self-hosted / cloud | ✓                  | ✓             | ✓      | ✓               | ✓     |
| Metal                    | cloud               | ✓                  |               | ✓      | ✓               |       |
| MongoDB Atlas            | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| MyScale                  | cloud               | ✓                  | ✓             | ✓      | ✓               |       |
| Milvus / Zilliz          | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| Neo4jVector              | self-hosted / cloud |                    |               | ✓      | ✓               |       |
| OpenSearch               | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| Pinecone                 | cloud               | ✓                  | ✓             | ✓      | ✓               |       |
| Postgres                 | self-hosted / cloud | ✓                  | ✓             | ✓      | ✓               | ✓     |
| pgvecto.rs               | self-hosted / cloud | ✓                  | ✓             | ✓      | ✓               |       |
| Qdrant                   | self-hosted / cloud | ✓                  | ✓             | ✓      | ✓               | ✓     |
| Redis                    | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| Simple                   | in-memory           | ✓                  |               | ✓      |                 |       |
| SingleStore              | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| Supabase                 | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| Tair                     | cloud               | ✓                  |               | ✓      | ✓               |       |
| TencentVectorDB          | cloud               | ✓                  | ✓             | ✓      | ✓               |       |
| Timescale                |                     | ✓                  |               | ✓      | ✓               | ✓     |
| Typesense                | self-hosted / cloud | ✓                  |               | ✓      | ✓               |       |
| Weaviate                 | self-hosted / cloud | ✓                  | ✓             | ✓      | ✓               |       |

For more details, see [Vector Store Integrations](/community/integrations/vector_stores.md).

```{toctree}
---
caption: Examples
maxdepth: 1
---
/examples/vector_stores/AstraDBIndexDemo.ipynb
/examples/vector_stores/AsyncIndexCreationDemo.ipynb
/examples/vector_stores/AzureCosmosDBMongoDBvCoreDemo.ipynb
/examples/vector_stores/CassandraIndexDemo.ipynb
/examples/vector_stores/ChromaIndexDemo.ipynb
/examples/vector_stores/CognitiveSearchIndexDemo.ipynb
/examples/vector_stores/DashvectorIndexDemo.ipynb
/examples/vector_stores/DeepLakeIndexDemo.ipynb
/examples/vector_stores/DocArrayHnswIndexDemo.ipynb
/examples/vector_stores/DocArrayInMemoryIndexDemo.ipynb
/examples/vector_stores/EpsillaIndexDemo.ipynb
/examples/vector_stores/JaguarIndexDemo.ipynb
/examples/vector_stores/LanceDBIndexDemo.ipynb
/examples/vector_stores/MetalIndexDemo.ipynb
/examples/vector_stores/MilvusIndexDemo.ipynb
/examples/vector_stores/MyScaleIndexDemo.ipynb
/examples/vector_stores/ElasticsearchIndexDemo.ipynb
/examples/vector_stores/FaissIndexDemo.ipynb
/examples/vector_stores/MongoDBAtlasVectorSearch.ipynb
/examples/vector_stores/Neo4jVectorDemo.ipynb
/examples/vector_stores/OpensearchDemo.ipynb
/examples/vector_stores/PineconeIndexDemo.ipynb
/examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb
/examples/vector_stores/PGVectoRsDemo.ipynb
/examples/vector_stores/postgres.ipynb
/examples/vector_stores/RedisIndexDemo.ipynb
/examples/vector_stores/QdrantIndexDemo.ipynb
/examples/vector_stores/qdrant_hybrid.ipynb
/examples/vector_stores/RocksetIndexDemo.ipynb
/examples/vector_stores/SimpleIndexDemo.ipynb
/examples/vector_stores/SupabaseVectorIndexDemo.ipynb
/examples/vector_stores/TairIndexDemo.ipynb
/examples/vector_stores/TencentVectorDBIndexDemo.ipynb
/examples/vector_stores/Timescalevector.ipynb
/examples/vector_stores/WeaviateIndexDemo.ipynb
/examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb
/examples/vector_stores/ZepIndexDemo.ipynb
```
