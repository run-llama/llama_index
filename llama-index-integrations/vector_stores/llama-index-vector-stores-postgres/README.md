# LlamaIndex Vector_Stores Integration: Postgres

This integration allows you to use PostgreSQL with the `pgvector` extension as a vector store for LlamaIndex.

## Installation

```bash
pip install llama-index-vector-stores-postgres
```

## Usage

### Basic Setup

```python
from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database="your_database",
    host="localhost",
    password="your_password",
    port="5432",
    user="your_user",
    table_name="your_table",
    embed_dim=1536,  # OpenAI embedding dimension
)
```

### Query Modes

The PGVectorStore supports multiple query modes:

- `DEFAULT` - Standard similarity search
- `HYBRID` - Combines dense and sparse retrieval
- `SPARSE` - BM25-based text search
- `TEXT_SEARCH` - Full-text search
- `MMR` - Maximal Marginal Relevance for diverse results

### MMR (Maximal Marginal Relevance) Queries

MMR balances relevance and diversity in search results. Use it when you want results that are both relevant to the query and diverse from each other.

```python
from llama_index.core import VectorStoreIndex

# Create index with PGVectorStore
index = VectorStoreIndex.from_vector_store(vector_store)

# Query engine with MMR
query_engine = index.as_query_engine(
    vector_store_query_mode="mmr",
    similarity_top_k=5,
    vector_store_kwargs={
        "mmr_threshold": 0.5,  # 0=max diversity, 1=max similarity
    },
)
response = query_engine.query("Your question here")

# Retriever with MMR
retriever = index.as_retriever(
    vector_store_query_mode="mmr",
    similarity_top_k=5,
    vector_store_kwargs={
        "mmr_threshold": 0.3,  # Lower = more diverse results
        "mmr_prefetch_factor": 4.0,  # Prefetch multiplier (default: 4.0)
    },
)
nodes = retriever.retrieve("Your query here")
```

#### MMR Parameters

| Parameter             | Description                                         | Default |
| --------------------- | --------------------------------------------------- | ------- |
| `mmr_threshold`       | Balance between relevance (1.0) and diversity (0.0) | 0.5     |
| `mmr_prefetch_factor` | Multiplier for candidate pool size                  | 4.0     |
| `mmr_prefetch_k`      | Exact candidate pool size (overrides factor)        | None    |
