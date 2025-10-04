# LlamaIndex Vector Stores Integration: ParadeDB

This module adds full ParadeDB integration enabling hybrid search with BM25 and vector similarity (HNSW) in PostgreSQL.

---

## Quick Setup

---

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

### Disclaimer

This integration was based on the Postgres Vector Store implementation:

**version = "0.5.5"**


However, **`customize_query_fn`** and other Postgres-specific query customization features are **not supported** in this ParadeDB version, as the focus here is on BM25 and hybrid retrieval.

Feel free to contribute and extend this module further.

