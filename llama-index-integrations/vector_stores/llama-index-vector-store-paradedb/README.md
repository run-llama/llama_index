# LlamaIndex Vector Stores Integration: ParadeDB

This module adds full ParadeDB integration enabling hybrid search with BM25 and vector similarity (HNSW) in PostgreSQL.

---
# Testing Setup

## Installation

First, install the package:

```bash
pip install llama-index-vector-stores-paradedb
```

## Database Setup

### 1. **Setup example**

Run ParadeDB locally:

```bash
docker run --name paradedb \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=mark90 \ 
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  -d paradedb/paradedb:latest
```

---

### 2. **Usage example**

```python
import os
from dotenv import load_dotenv
from sqlalchemy import make_url
from llama_index.vector_stores.paradedb import ParadeDBVectorStore

def get_vector_store(table_name: str = "pgvector") -> ParadeDBVectorStore:
    """
    Creates and returns a new ParadeDBVectorStore instance using environment variables.
    """
    load_dotenv()

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_DATABASE")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}"
    url = make_url(connection_string)

    return ParadeDBVectorStore.from_params(
        database=database,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table_name,
        text_search_config="english",
        hybrid_search=True,  # needed to use bm25
        use_bm25=True,
        embed_dim=int(os.getenv("EMBEDDING_DIM")),
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
```

---

### Notes

* Set `hybrid_search=True` and `use_bm25=True` to enable **hybrid BM25 + vector** retrieval.
* You **must** use the `paradedb/paradedb:latest` image — not `pgvector/pgvector`.
* The default schema name is `paradedb` to enable BM25.
* Fully compatible with **llama-index-core** and other vector store interfaces.

---

### Results

The following results demonstrate the difference between BM25 and TSVECTOR ranking methods:

| Method | Rank | Node ID | Score |
|--------|------|---------|--------|
| TSVECTOR | Top1 | ccc | 0.060793 |
| TSVECTOR | Top2 | ddd | 0.060793 |
| BM25 | Top1 | ddd | 0.678537 |
| BM25 | Top2 | ccc | 0.507418 |

**Key observations**:
- BM25 produces higher similarity scores overall
- BM25 shows more differentiation between results (0.678 vs 0.507)
- TSVECTOR gives equal scores to both results (0.060793)
- BM25 ranks 'ddd' higher than 'ccc', while TSVECTOR treats them equally