# LlamaIndex ZeusDB Integration

ZeusDB vector database integration for LlamaIndex. Connect LlamaIndex's RAG framework with high-performance, enterprise-grade vector database.

## Features

- **Production Ready**: Built for enterprise-scale RAG applications
- **Persistence**: Complete save/load functionality with cross-platform compatibility
- **Advanced Filtering**: Comprehensive metadata filtering with complex operators
- **MMR Support**: Maximal Marginal Relevance for diverse, non-redundant results
- **Quantization**: Product Quantization (PQ) for memory-efficient vector storage
- **Async Support**: Async wrappers for non-blocking operations (`aadd`, `aquery`, `adelete_nodes`)

## Installation

```bash
pip install llama-index-vector-stores-zeusdb
```

## Quick Start

```python
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Set up embedding model and LLM
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-5")

# Create ZeusDB vector store
vector_store = ZeusDBVectorStore(
    dim=1536,  # OpenAI embedding dimension
    distance="cosine",
    index_type="hnsw"
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create documents
documents = [
    Document(text="ZeusDB is a high-performance vector database."),
    Document(text="LlamaIndex provides RAG capabilities."),
    Document(text="Vector search enables semantic similarity.")
]

# Create index and store documents
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is ZeusDB?")
print(response)
```

## Advanced Features

### Persistence

Save and load indexes with complete state preservation:

```python
# Save index to disk
vector_store.save_index("my_index.zdb")

# Load index from disk
loaded_store = ZeusDBVectorStore.load_index("my_index.zdb")
```

### MMR Search

Balance relevance and diversity for comprehensive results:

```python
from llama_index.core.vector_stores.types import VectorStoreQuery

# Query with MMR for diverse results
query_embedding = embed_model.get_text_embedding("your query")
results = vector_store.query(
    VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5),
    mmr=True,
    fetch_k=20,
    mmr_lambda=0.7  # 0.0=max diversity, 1.0=pure relevance
)

# Note: MMR automatically enables return_vector=True for diversity calculation
# Results contain ids and similarities (nodes=None)
```

### Quantization

Reduce memory usage with Product Quantization:

```python
vector_store = ZeusDBVectorStore(
    dim=1536,
    distance="cosine",
    quantization_config={
        'type': 'pq',
        'subvectors': 8,
        'bits': 8,
        'training_size': 1000,
        'storage_mode': 'quantized_only'
    }
)
```

### Async Operations

Non-blocking operations for web servers and concurrent workflows:

```python
import asyncio

# Async add
node_ids = await vector_store.aadd(nodes)

# Async query
results = await vector_store.aquery(query_obj)

# Async delete
await vector_store.adelete_nodes(node_ids=["id1", "id2"])
```

### Metadata Filtering

Filter results by metadata:

```python
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    FilterOperator,
    FilterCondition
)

# Create metadata filter
filters = MetadataFilters.from_dicts([
    {"key": "category", "value": "tech", "operator": FilterOperator.EQ},
    {"key": "year", "value": 2024, "operator": FilterOperator.GTE}
], condition=FilterCondition.AND)

# Query with filters
results = vector_store.query(
    VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        filters=filters
    )
)
```

**Supported operators**: EQ, NE, GT, GTE, LT, LTE, IN, NIN, ANY, ALL, CONTAINS, TEXT_MATCH, TEXT_MATCH_INSENSITIVE

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dim` | Vector dimension | Required |
| `distance` | Distance metric (`cosine`, `l2`, `l1`) | `cosine` |
| `index_type` | Index type (`hnsw`) | `hnsw` |
| `m` | HNSW connectivity parameter | 16 |
| `ef_construction` | HNSW build-time search depth | 200 |
| `expected_size` | Expected number of vectors | 10000 |
| `quantization_config` | PQ quantization settings | None |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
