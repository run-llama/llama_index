# Neo4j Query Engine Pack

This LlamaPack creates a Neo4j query engine, and executes its `query` function. This pack offers the option of creating multiple types of query engines, namely:

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
llamaindex-cli download-llamapack Neo4jQueryEnginePack --download-dir ./neo4j_pack
```

You can then inspect the files at `./neo4j_pack` and use them as a template for your own project!

## Code Usage

You can download the pack to a `./neo4j_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
Neo4jQueryEnginePack = download_llama_pack(
    "Neo4jQueryEnginePack", "./neo4j_pack"
)
```

From here, you can use the pack, or inspect and modify the pack in `./neo4j_pack`.

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

# get Neo4j credentials (assume it's stored in credentials.json)
with open("credentials.json") as f:
    neo4j_connection_params = json.load(f)
    username = neo4j_connection_params["username"]
    password = neo4j_connection_params["password"]
    url = neo4j_connection_params["url"]
    database = neo4j_connection_params["database"]

# create the pack
neo4j_pack = Neo4jQueryEnginePack(
    username=username, password=password, url=url, database=database, docs=docs
)
```

Optionally, you can pass in the `query_engine_type` from `Neo4jQueryEngineType` to construct `Neo4jQueryEnginePack`. If `query_engine_type` is not defined, it defaults to Knowledge Graph vector based entity retrieval.

```python
from llama_index.core.packs.neo4j_query_engine.base import Neo4jQueryEngineType

# create the pack
neo4j_pack = Neo4jQueryEnginePack(
    username=username,
    password=password,
    url=url,
    database=database,
    docs=docs,
    query_engine_type=Neo4jQueryEngineType.KG_HYBRID,
)
```

`Neo4jQueryEnginePack` is a enum defined as follows:

```python
class Neo4jQueryEngineType(str, Enum):
    """Neo4j query engine type"""

    KG_KEYWORD = "keyword"
    KG_HYBRID = "hybrid"
    RAW_VECTOR = "vector"
    RAW_VECTOR_KG_COMBO = "vector_kg"
    KG_QE = "KnowledgeGraphQueryEngine"
    KG_RAG_RETRIEVER = "KnowledgeGraphRAGRetriever"
```

The `run()` function is a light wrapper around `query_engine.query()`, see a sample query below.

```python
response = neo4j_pack.run("Tell me about the benefits of paleo diet.")
```

You can also use modules individually.

```python
# call the query_engine.query()
query_engine = neo4j_pack.query_engine
response = query_engine.query("query_str")
```
