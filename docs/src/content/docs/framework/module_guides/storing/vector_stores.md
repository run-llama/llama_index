---
title: Vector Stores
---

Vector stores contain embedding vectors of ingested document chunks
(and sometimes the document chunks as well).

## Simple Vector Store

By default, LlamaIndex uses a simple in-memory vector store that's great for quick experimentation.
They can be persisted to (and loaded from) disk by calling `vector_store.persist()` (and `SimpleVectorStore.from_persist_path(...)` respectively).

## Vector Store Options & Feature Support

LlamaIndex supports over 20 different vector store options.
We are actively adding more integrations and improving feature coverage for each.

| Vector Store               | Type                    | Metadata Filtering | Hybrid Search | Delete | Store Documents | Async                         |
| -------------------------- | ----------------------- | ------------------ | ------------- | ------ | --------------- | ----------------------------- |
| Alibaba Cloud OpenSearch   | cloud                   | ✓                  |               | ✓      | ✓               | ✓                             |
| Apache Cassandra®         | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |                               |
| Astra DB                   | cloud                   | ✓                  |               | ✓      | ✓               |                               |
| Azure AI Search            | cloud                   | ✓                  | ✓             | ✓      | ✓               |                               |
| Azure CosmosDB Mongo vCore | cloud                   |                    |               | ✓      | ✓               |                               |
| Azure CosmosDB NoSql       | cloud                   |                    |               | ✓      | ✓               |                               |
| BaiduVectorDB              | cloud                   | ✓                  | ✓             |        | ✓               |                               |
| ChatGPT Retrieval Plugin   | aggregator              |                    |               | ✓      | ✓               |                               |
| Chroma                     | self-hosted             | ✓                  |               | ✓      | ✓               |                               |
| Couchbase                  | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |                               |
| DashVector                 | cloud                   | ✓                  | ✓             | ✓      | ✓               |                               |
| Databricks                 | cloud                   | ✓                  |               | ✓      | ✓               |                               |
| Deeplake                   | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |                               |
| DocArray                   | aggregator              | ✓                  |               | ✓      | ✓               |                               |
| DuckDB                     | in-memory / self-hosted | ✓                  |               | ✓      | ✓               |                               |
| DynamoDB                   | cloud                   |                    |               | ✓      |                 |                               |
| Elasticsearch              | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓                             |
| FAISS                      | in-memory               |                    |               |        |                 |                               |
| Google AlloyDB             | cloud                   | ✓                  |               | ✓      | ✓               | ✓                             |
| Google Cloud SQL Postgres  | cloud                   | ✓                  |               | ✓      | ✓               | ✓                             |
| Hnswlib                    | in-memory               |                    |               |        |                 |                               |
| txtai                      | in-memory               |                    |               |        |                 |                               |
| Jaguar                     | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |                               |
| LanceDB                    | cloud                   | ✓                  |               | ✓      | ✓               |                               |
| Lantern                    | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓                             |
| MongoDB Atlas              | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |                               |
| MyScale                    | cloud                   | ✓                  | ✓             | ✓      | ✓               |                               |
| Milvus / Zilliz            | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |                               |
| Neo4jVector                | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |                               |
| OpenSearch                 | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓                             |
| Pinecone                   | cloud                   | ✓                  | ✓             | ✓      | ✓               |                               |
| Postgres                   | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓                             |
| pgvecto.rs                 | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |                               |
| Qdrant                     | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               | ✓                             |
| Redis                      | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |                               |
| S3                         | cloud                   | ✓                  |               | ✓      | ✓               | ✓\* (using asyncio.to_thread) |
| Simple                     | in-memory               | ✓                  |               | ✓      |                 |                               |
| SingleStore                | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |                               |
| Supabase                   | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |                               |
| Tablestore                 | cloud                   | ✓                  | ✓             | ✓      | ✓               |                               |
| Tair                       | cloud                   | ✓                  |               | ✓      | ✓               |                               |
| TiDB                       | cloud                   | ✓                  |               | ✓      | ✓               |                               |
| TencentVectorDB            | cloud                   | ✓                  | ✓             | ✓      | ✓               |                               |
| Timescale                  |                         | ✓                  |               | ✓      | ✓               | ✓                             |
| Typesense                  | self-hosted / cloud     | ✓                  |               | ✓      | ✓               |                               |
| Upstash                    | cloud                   |                    |               |        | ✓               |                               |
| VectorX DB                 | cloud                   | ✓                  | ✓             | ✓      | ✓               | ✓                             |
| Vearch                     | self-hosted             | ✓                  |               | ✓      | ✓               |                               |
| Vespa                      | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |                               |
| Vertex AI Vector Search    | cloud                   | ✓                  |               | ✓      | ✓               |                               |
| Weaviate                   | self-hosted / cloud     | ✓                  | ✓             | ✓      | ✓               |                               |
| WordLift                   | cloud                   | ✓                  | ✓             | ✓      | ✓               | ✓                             |

For more details, see [Vector Store Integrations](/python/framework/community/integrations/vector_stores).

## Example Notebooks

- [Alibaba Cloud OpenSearch](/python/examples/vector_stores/alibabacloudopensearchindexdemo)
- [Astra DB](/python/examples/vector_stores/astradbindexdemo)
- [Async Index Creation](/python/examples/vector_stores/asyncindexcreationdemo)
- [Azure AI Search](/python/examples/vector_stores/azureaisearchindexdemo)
- [Azure Cosmos DB Mongo vCore](/python/examples/vector_stores/azurecosmosdbmongodbvcoredemo)
- [Azure Cosmos DB NoSql](/python/examples/vector_stores/azurecosmosdbnosqldemo)
- [Baidu](/python/examples/vector_stores/baiduvectordbindexdemo)
- [Caasandra](/python/examples/vector_stores/cassandraindexdemo)
- [Chromadb](/python/examples/vector_stores/chromaindexdemo)
- [Couchbase](/python/examples/vector_stores/couchbasevectorstoredemo)
- [Dash](/python/examples/vector_stores/dashvectorindexdemo)
- [Databricks](/python/examples/vector_stores/databricksvectorsearchdemo)
- [Deeplake](/python/examples/vector_stores/deeplakeindexdemo)
- [DocArray HNSW](/python/examples/vector_stores/docarrayhnswindexdemo)
- [DocArray in-Memory](/python/examples/vector_stores/docarrayinmemoryindexdemo)
- [DuckDB](/python/examples/vector_stores/duckdbdemo)
- [Espilla](/python/examples/vector_stores/epsillaindexdemo)
- [Google AlloyDB for PostgreSQL](/python/examples/vector_stores/alloydbvectorstoredemo)
- [Google Cloud SQL for PostgreSQL](/python/examples/vector_stores/cloudsqlpgvectorstoredemo)
- [Jaguar](/python/examples/vector_stores/jaguarindexdemo)
- [LanceDB](/python/examples/vector_stores/lancedbindexdemo)
- [Lantern](/python/examples/vector_stores/lanternindexdemo)
- [Milvus](/python/examples/vector_stores/milvusindexdemo)
- [Milvus Async API](/python/examples/vector_stores/milvusasyncapidemo)
- [Milvus Full-Text Search](/python/examples/vector_stores/milvusfulltextsearchdemo)
- [Milvus Hybrid Search](/python/examples/vector_stores/milvushybridindexdemo)
- [MyScale](/python/examples/vector_stores/myscaleindexdemo)
- [ElasticSearch](/python/examples/vector_stores/elasticsearchindexdemo)
- [FAISS](/python/examples/vector_stores/faissindexdemo)
- [Hnswlib](/python/examples/vector_stores/hnswlibindexdemo)
- [MongoDB Atlas](/python/examples/vector_stores/mongodbatlasvectorsearch)
- [Neo4j](/python/examples/vector_stores/neo4jvectordemo)
- [OpenSearch](/python/examples/vector_stores/opensearchdemo)
- [Pinecone](/python/examples/vector_stores/pineconeindexdemo)
- [Pinecone Hybrid Search](/python/examples/vector_stores/pineconeindexdemo-hybrid)
- [PGvectoRS](/python/examples/vector_stores/pgvectorsdemo)
- [Postgres](/python/examples/vector_stores/postgres)
- [Redis](/python/examples/vector_stores/redisindexdemo)
- [Qdrant](/python/examples/vector_stores/qdrantindexdemo)
- [Qdrant Hybrid Search](/python/examples/vector_stores/qdrant_hybrid)
- [Rockset](/python/examples/vector_stores/rocksetindexdemo)
- [S3](/python/examples/vector_stores/s3vectorstore)
- [Simple](/python/examples/vector_stores/simpleindexdemo)
- [Supabase](/python/examples/vector_stores/supabasevectorindexdemo)
- [Tablestore](/python/examples/vector_stores/tablestoredemo)
- [Tair](/python/examples/vector_stores/tairindexdemo)
- [TiDB](/python/examples/vector_stores/tidbvector)
- [Tencent](/python/examples/vector_stores/tencentvectordbindexdemo)
- [Timesacle](/python/examples/vector_stores/timescalevector)
- [Upstash](/python/examples/vector_stores/upstashvectordemo)
- [VectorX DB](/python/examples/vector_stores/vectorxdbdemo)
- [Vearch](/python/examples/vector_stores/vearchdemo)
- [Vespa](/python/examples/vector_stores/vespaindexdemo)
- [Vertex AI Vector Search](/python/examples/vector_stores/vertexaivectorsearchdemo)
- [Weaviate](/python/examples/vector_stores/weaviateindexdemo)
- [Weaviate Hybrid Search](/python/examples/vector_stores/weaviateindexdemo-hybrid)
- [WordLift](/python/examples/vector_stores/wordliftdemo)
- [Zep](/python/examples/vector_stores/zepindexdemo)
