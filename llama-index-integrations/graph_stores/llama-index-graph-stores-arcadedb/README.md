# LlamaIndex Graph Store Integration: ArcadeDB

`llama-index-graph-stores-arcadedb` provides an **ArcadeDB** backend for LlamaIndex's `PropertyGraphIndex`.

ArcadeDB is a multi-model database that speaks the **Bolt protocol** natively and supports **OpenCypher** queries. This integration uses the standard `neo4j` Python driver — no APOC required.

## Installation

```bash
pip install llama-index-graph-stores-arcadedb
```

## Quick Start

```python
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.arcadedb import ArcadeDBPropertyGraphStore

graph_store = ArcadeDBPropertyGraphStore(
    url="bolt://localhost:7687",
    username="root",
    password="playwithdata",
    database="mydb",
)

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
)

# Query
retriever = index.as_retriever(include_text=False)
results = retriever.retrieve("What is …?")
```

## Features

| Feature | Supported |
|---|---|
| Structured (Cypher) queries | Yes |
| Vector similarity search | Yes (native HNSW) |
| Schema introspection | Yes (sample-based) |
| Bolt protocol | Yes (native) |

## Running ArcadeDB

```bash
docker run -d --name arcadedb \
  -p 2480:2480 -p 2424:2424 -p 7687:7687 \
  -e JAVA_OPTS="-Darcadedb.server.plugins=Bolt:com.arcadedb.bolt.BoltProtocolPlugin" \
  arcadedata/arcadedb:latest
```

Create a database:

```bash
curl -X POST http://localhost:2480/api/v1/server \
  -d '{"command":"CREATE DATABASE mydb"}' \
  -u root:playwithdata
```

## Configuration

| Parameter | Env var | Default |
|---|---|---|
| `url` | `ARCADEDB_BOLT_URL` | `bolt://localhost:7687` |
| `username` | `ARCADEDB_USERNAME` | `root` |
| `password` | `ARCADEDB_PASSWORD` | `playwithdata` |
| `database` | `ARCADEDB_DATABASE` | `""` |

## Design Notes

ArcadeDB enforces a **single type per vertex**. This integration stores entities as `Entity` vertices with the semantic label (e.g. `PERSON`, `CITY`) in the `label` property, and text chunks as `Chunk` vertices.

All APOC procedures used by the Neo4j integration are replaced with pure Cypher equivalents. Variable-length path matching (`MATCH (n)-[*1..depth]-()`) replaces `apoc.path.expand`.
