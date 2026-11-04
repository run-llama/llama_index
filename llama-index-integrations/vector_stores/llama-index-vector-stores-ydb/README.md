# LlamaIndex Vector_Stores Integration: YDB

Use [YDB](https://ydb.tech) as a vector store for LlamaIndex.

## Installation

```bash
pip install llama-index-vector-stores-ydb
```

## Usage

```python
from llama_index.vector_stores.ydb import YDBVectorStore

vector_store = YDBVectorStore(
    host="localhost",
    port=2136,
    database="/local",
    table_name="my_nodes",
    embed_dim=1536,
)
```

Current limitations: exact (full scan) cosine search, `DEFAULT` query mode only, no metadata filters.
