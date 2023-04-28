# Vector Stores

Vector stores contain embedding vectors of ingested document chunks 
(and sometimes the document chunks as well).

## Simple Vector Store
By default, LlamaIndex uses a simple in-memory vector store that's great for quick experimentation.
They can be persisted to (and loaded from) disk by calling `vector_store.persist()` (and `SimpleVectorStore.from_persist_path(...)` respectively).

## Third-Party Vector Store Integrations
We also integrate with a wide range of vector store implementations. 
They mainly differ in 2 aspects:
1. in-memory vs. hosted
2. stores only vector embeddings vs. also stores documents

### In-Memory Vector Stores
* Faiss
* Chroma

### (Self) Hosted Vector Stores
* Pinecone
* Weaviate
* Milvus/Zilliz
* Qdrant
* Chroma
* Opensearch
* DeepLake
* MyScale

### Others
* ChatGPTRetrievalPlugin

For more details, see [Vector Store Integrations](/how_to/integrations/vector_stores.md).
