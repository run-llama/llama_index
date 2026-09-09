# LlamaIndex Vector Store Integration: Infino

[Infino](https://pypi.org/project/infino/) is a retrieval engine that stores your
data as Apache Parquet on object storage and runs SQL, full-text (BM25), vector,
and hybrid search over it from a single **embedded** library — no server, no
cluster, no separate index service. `InfinoVectorStore` maps LlamaIndex onto one Infino
table that holds the node id, the node text, the embedding, filterable metadata,
and a JSON catch-all together.

Because `stores_text=True`, that one table replaces **both** the vector store and
the docstore: `VectorStoreIndex.from_vector_store(store)` rebuilds full nodes with
no separate document store to run.

```bash
pip install llama-index-vector-stores-infino
```

## Quickstart — embedded, no server

Point it at a local directory. Nothing to run, nothing to connect to.

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.infino import InfinoVectorStore

store = InfinoVectorStore("/data/my_index", table_name="docs")

# Add nodes that already carry embeddings.
store.add(
    [
        TextNode(text="the quick brown fox", embedding=[...]),
        TextNode(text="electric vehicles charge overnight", embedding=[...]),
    ]
)

# One table holds text + vectors + metadata — no docstore needed.
index = VectorStoreIndex.from_vector_store(store, embed_model=my_embed_model)
nodes = index.as_retriever(similarity_top_k=3).retrieve("fast animals")
```

## Straight over object storage (still no server)

Swap the local path for a bucket URI and the same store runs directly over S3,
GCS, or Azure Blob — the data lives as Parquet in your bucket, queried in place.

```python
store = InfinoVectorStore(
    "s3://my-bucket/rag-index",
    table_name="docs",
    storage_options={"aws_region": "us-east-1"},  # + credentials
)
```

## Hosted Infino Cloud

The same class connects to [Infino Cloud](https://platform.infino.ws) with a URL
and an API key — no code change beyond the constructor. Sign up at
**[platform.infino.ws](https://platform.infino.ws)** to get an API key, then point
the store at the data-plane URL for your database
(`https://api.platform.infino.ws/<database>`):

```python
store = InfinoVectorStore(
    "https://api.platform.infino.ws/your-database",
    table_name="docs",
    api_key="…",              # or api_key_file="~/.infino/key"
)
```

Because LlamaIndex supplies the embeddings, the hosted mode writes and queries the
platform's data plane directly — there is no server-side embedding step to
configure. Start local, and switch to the cloud when you outgrow one machine by
changing only the `uri` and adding a key.

## Hybrid search (BM25 + vector in one call)

```python
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)

result = store.query(
    VectorStoreQuery(
        query_str="electric vehicles",
        query_embedding=[...],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.HYBRID,
    )
)
```

`mode=VectorStoreQueryMode.TEXT_SEARCH` runs pure BM25; the default mode runs
vector search.

## Metadata filtering

Promote the metadata keys you want to filter on to real, typed columns at
construction; the rest are stored (and returned) in a JSON catch-all.

```python
import pyarrow as pa
from llama_index.core.vector_stores.types import (
    MetadataFilter, MetadataFilters, FilterOperator, VectorStoreQuery,
)

store = InfinoVectorStore(
    "/data/my_index",
    metadata_columns=["category", pa.field("year", pa.int64())],
)

store.query(
    VectorStoreQuery(
        query_embedding=[...],
        similarity_top_k=5,
        filters=MetadataFilters(
            filters=[MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)]
        ),
    )
)
```

## SQL over the same table

Infino also runs SQL over the very same table, so you can do analytics on your RAG
data — counts, `GROUP BY`, top-N — that a pure vector store can't. It is available
through the underlying connection (`store.client`), alongside the LlamaIndex
retrieval API:

```python
rows = store.client.query_sql(
    "SELECT category, COUNT(*) AS n FROM docs GROUP BY category ORDER BY n DESC"
)
```

The BM25, vector, and hybrid search functions are also usable as SQL table
functions, so relevance ranking and aggregation compose in one query.

## Feature support

| metadata filter | hybrid (BM25+vector) | delete | stores_text | async |
| :-------------: | :------------------: | :----: | :---------: | :---: |
| ✓ | ✓ | ✓ | ✓ | ✓ |

## Performance and feature richness

Infino is a full retrieval engine, not a single-purpose vector index. One embedded
library gives you **vector, full-text (BM25), hybrid, and SQL** over the same data,
with **metadata filtering**, **deletes**, and snapshot-isolated reads — all over
ordinary **Apache Parquet** that any other Parquet tool can still read. So a single
`InfinoVectorStore` covers retrieval patterns that usually take a vector database
*plus* a search engine *plus* a warehouse.

On performance, Infino keeps data as Parquet on object storage and caches the hot
working set in RAM and on local disk, serving low-latency vector, BM25, and hybrid
queries while keeping **storage cheap and decoupled from compute** — no cluster to
size and no fully memory-resident index to pay for. It is benchmarked on the
standard vector suites and is competitive on recall and throughput while being
substantially cheaper to run at scale. The same store scales from a laptop
directory to object storage to a hosted endpoint without changing your code.

## Running the tests

The tests run entirely in embedded local mode over a temp directory — no model,
no network:

```bash
pip install llama-index-core pytest pytest-asyncio pyarrow infino
pytest tests
```
