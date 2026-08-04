# LlamaIndex Graph Stores Integration: FalkorDB

[FalkorDB](https://www.falkordb.com/) is a fast, low-latency property graph database
built on a sparse-matrix engine and queried with Cypher. This package provides two
integrations:

- `FalkorDBPropertyGraphStore` — the modern `PropertyGraphStore` used by `PropertyGraphIndex`
- `FalkorDBGraphStore` — the legacy triplet-based `GraphStore`

## Installation

```bash
pip install llama-index-graph-stores-falkordb
```

## Running FalkorDB

```bash
docker run -p 6379:6379 -p 3000:3000 -v $PWD/data:/data falkordb/falkordb:latest
```

The browser UI is then available on <http://localhost:3000>. A managed instance is
also available on [FalkorDB Cloud](https://app.falkordb.cloud/).

## Usage

```python
from llama_index.core import Document
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore

graph_store = FalkorDBPropertyGraphStore(
    url="redis://localhost:6379",
    database="falkor",  # the graph name
)

index = PropertyGraphIndex.from_documents(
    [Document(text="Alice works for Acme since 2023.")],
    property_graph_store=graph_store,
)

retriever = index.as_retriever(include_text=True)
print(retriever.retrieve("Who does Alice work for?"))
```

The store is also a context manager, so the connection is released deterministically:

```python
with FalkorDBPropertyGraphStore(url="redis://localhost:6379") as graph_store:
    graph_store.upsert_nodes([...])
```

### Connecting to a secured instance

Any additional keyword argument is forwarded to the FalkorDB client:

```python
graph_store = FalkorDBPropertyGraphStore(
    url="redis://my-instance.falkordb.cloud:6379",
    username="falkordb",
    password="...",
    ssl=True,
    socket_timeout=30,
)
```

## Configuration

| Argument                | Default    | Description                                                             |
| ----------------------- | ---------- | ----------------------------------------------------------------------- |
| `url`                   | —          | Connection URL, e.g. `redis://localhost:6379`.                          |
| `database`              | `"falkor"` | Name of the graph to read from and write to.                            |
| `refresh_schema`        | `True`     | Read the graph schema on startup.                                       |
| `sanitize_query_output` | `True`     | Strip oversized values (such as embeddings) from query results.         |
| `create_indexes`        | `True`     | Create the range and vector indexes used for lookups and vector search. |
| `timeout`               | `None`     | Per-query timeout in milliseconds.                                      |

### Indexes

With `create_indexes=True` (the default) the store creates a range index on
`__Entity__.id` and `Chunk.id`, which is what makes ingestion `MERGE`s and `id`
lookups fast. A vector index on `__Entity__.embedding` is created lazily, the first
time a node with an embedding is written, using that embedding's dimension and
cosine similarity. `vector_query` uses the index when no metadata filters are
supplied and falls back to an exact scan otherwise.

If the graph already contains a vector index with a different dimension, the store
logs a warning and falls back to the exact scan — recreate the index (or the graph)
when you change embedding model.

### Schema

`refresh_schema()` enumerates every label and relationship type in the graph, then
samples up to 1000 nodes per label to infer property types, and up to 1000
relationships per type to infer which labels they connect. Labels and relationship
types are therefore always complete, while the reported property _types_ and
`(:Start)-[:REL]->(:End)` pairs are a best-effort sample — a combination that occurs
in fewer than 1 in 1000 relationships of its type may be missed. The `embedding`
property is excluded so it never reaches the text-to-Cypher prompt.

Sampling matters because `PropertyGraphIndex` refreshes the schema after **every**
ingestion batch; traversing all relationships instead would make ingestion cost grow
with the size of the graph.

### Ingestion performance

Every read this store issues is scoped to `__Entity__` or `Chunk` so FalkorDB's
label-scoped range indexes apply. This is what keeps the per-batch deduplication that
`PropertyGraphIndex` performs (`get()` twice, plus a schema refresh) from degrading
into full-graph scans as the graph grows.

## Legacy graph store

```python
from llama_index.graph_stores.falkordb import FalkorDBGraphStore

graph_store = FalkorDBGraphStore(url="redis://localhost:6379")
graph_store.upsert_triplet("Alice", "works_for", "Acme")
```

## Testing

The tests start a throwaway FalkorDB container automatically (Docker required):

```bash
pytest tests
```

Set `FALKORDB_TEST_URL` to run them against an already running instance instead:

```bash
FALKORDB_TEST_URL=redis://localhost:6379 pytest tests
```
