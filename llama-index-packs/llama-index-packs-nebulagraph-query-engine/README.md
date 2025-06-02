# NebulaGraph Query Engine Pack

This LlamaPack creates a NebulaGraph query engine, and executes its `query` function. This pack offers the option of creating multiple types of query engines, namely:

- Knowledge graph vector-based entity retrieval (default if no query engine type option is provided)
- Knowledge graph keyword-based entity retrieval
- Knowledge graph hybrid entity retrieval
- Raw vector index retrieval
- Custom combo query engine (vector similarity + KG entity retrieval)
- KnowledgeGraphQueryEngine
- KnowledgeGraphRAGRetriever

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack NebulaGraphQueryEnginePack --download-dir ./nebulagraph_pack
```

You can then inspect the files at `./nebulagraph_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./nebulagraph_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
NebulaGraphQueryEnginePack = download_llama_pack(
    "NebulaGraphQueryEnginePack", "./nebulagraph_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./nebulagraph_pack`.

Then, you can set up the pack like so:

```bash
pip install llama-index-readers-wikipedia
```

```python
# Load the docs (example of Paleo diet from Wikipedia)

from llama_index.readers.wikipedia import WikipediaReader

loader = WikipediaReader()
docs = loader.load_data(pages=["Paleolithic diet"], auto_suggest=False)
print(f"Loaded {len(docs)} documents")

# get NebulaGraph credentials (assume it's stored in credentials.json)
with open("credentials.json") as f:
    nebulagraph_connection_params = json.load(f)
    username = nebulagraph_connection_params["username"]
    password = nebulagraph_connection_params["password"]
    ip_and_port = nebulagraph_connection_params["ip_and_port"]

space_name = "paleo_diet"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]
max_triplets_per_chunk = 10

# create the pack
nebulagraph_pack = NebulaGraphQueryEnginePack(
    username=username,
    password=password,
    ip_and_port=ip_and_port,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    max_triplets_per_chunk=max_triplets_per_chunk,
    docs=docs,
)
```

Optionally, you can pass in the `query_engine_type` from `NebulaGraphQueryEngineType` to construct `NebulaGraphQueryEnginePack`. If `query_engine_type` is not defined, it defaults to Knowledge Graph vector based entity retrieval.

```python
from llama_index.core.packs.nebulagraph_query_engine.base import (
    NebulaGraphQueryEngineType,
)

# create the pack
nebulagraph_pack = NebulaGraphQueryEnginePack(
    username=username,
    password=password,
    ip_and_port=ip_and_port,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    max_triplets_per_chunk=max_triplets_per_chunk,
    docs=docs,
    query_engine_type=NebulaGraphQueryEngineType.KG_HYBRID,
)
```

`NebulaGraphQueryEnginePack` is a enum defined as follows:

```python
class NebulaGraphQueryEngineType(str, Enum):
    """NebulaGraph query engine type"""

    KG_KEYWORD = "keyword"
    KG_HYBRID = "hybrid"
    RAW_VECTOR = "vector"
    RAW_VECTOR_KG_COMBO = "vector_kg"
    KG_QE = "KnowledgeGraphQueryEngine"
    KG_RAG_RETRIEVER = "KnowledgeGraphRAGRetriever"
```

The `run()` function is a light wrapper around `query_engine.query()`, see a sample query below.

```python
response = nebulagraph_pack.run("Tell me about the benefits of paleo diet.")
```

You can also use modules individually.

```python
# call the query_engine.query()
query_engine = nebulagraph_pack.query_engine
response = query_engine.query("query_str")
```
