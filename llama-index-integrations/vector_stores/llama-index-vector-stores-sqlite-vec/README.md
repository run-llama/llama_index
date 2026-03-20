# LlamaIndex Vector Store Integration: sqlite-vec

A lightweight vector store integration using [sqlite-vec](https://github.com/asg017/sqlite-vec), a SQLite extension for vector similarity search.

## Installation

```bash
pip install llama-index-vector-stores-sqlite-vec
```

## Usage

```python
from llama_index.vector_stores.sqlite_vec import SqliteVecVectorStore

# In-memory
vector_store = SqliteVecVectorStore(embed_dim=1536)

# Persist to disk
vector_store = SqliteVecVectorStore(
    database_path="./vectors.db",
    embed_dim=1536,
)
```
