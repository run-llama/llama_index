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

```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/vector_stores/SimpleIndexDemo.ipynb
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
../../examples/vector_stores/RedisIndexDemo.ipynb
../../examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb
../../examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb
../../examples/vector_stores/AsyncIndexCreationDemo.ipynb
../../examples/vector_stores/SupabaseVectorIndexDemo.ipynb
```

