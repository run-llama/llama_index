向量存储
向量存储包含摄取文档块的嵌入向量（有时也包括文档块）。

## 简单向量存储
默认情况下，LlamaIndex使用一个简单的内存向量存储，非常适合快速实验。它们可以通过调用`vector_store.persist()`（以及`SimpleVectorStore.from_persist_path(...)`）持久化到（并从）磁盘中加载。

## 第三方向量存储集成
我们还集成了各种向量存储实现。它们主要在两个方面不同：
1. 内存 vs. 托管
2. 仅存储向量嵌入 vs. 也存储文档

### 内存向量存储
* Faiss
* Chroma

###（自）托管向量存储
* Pinecone
* Weaviate
* Milvus/Zilliz
* Qdrant
* Chroma
* Opensearch
* DeepLake
* MyScale

### 其他
* ChatGPTRetrievalPlugin

有关详细信息，请参阅[向量存储集成](/how_to/integrations/vector_stores.md)。