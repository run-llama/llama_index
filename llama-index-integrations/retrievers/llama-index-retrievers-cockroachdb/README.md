# LlamaIndex Retrievers Integration: CockroachDB

Standalone `BaseRetriever` wrapping [`llama-index-vector-stores-cockroachdb`](../../vector_stores/llama-index-vector-stores-cockroachdb) with knobs for C-SPANN beam-size tuning and MMR. Use this when you want CRDB-backed vector retrieval outside `VectorStoreIndex`, for example in a custom query pipeline.

## Installation

```bash
pip install llama-index-retrievers-cockroachdb
```

(`llama-index-vector-stores-cockroachdb` is pulled in as a dependency.)

## Usage

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.cockroachdb import CockroachDBRetriever
from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore

store = CockroachDBVectorStore.from_params(
    host="localhost", port=26257, user="root", password="",
    database="defaultdb", table_name="my_index",
    embed_dim=1536, distance_metric="cosine",
    sslmode="disable",
)

retriever = CockroachDBRetriever(
    vector_store=store,
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    similarity_top_k=5,
    vector_search_beam_size=128,
)

for node in retriever.retrieve("How does C-SPANN compare to HNSW?"):
    print(f"{node.score:.4f}  {node.node.get_content()[:80]}")
```

`mmr_threshold`, `mmr_prefetch_factor`, `mmr_prefetch_k`, and `filters` are also supported on the constructor.
